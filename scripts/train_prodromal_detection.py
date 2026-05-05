"""질환 발병 전 이상징후 감지 모델 학습 및 추론 스크립트.

실행:
    python scripts/train_prodromal_detection.py
    python scripts/train_prodromal_detection.py --config configs/prodromal.yaml --output-dir outputs/prodromal

4단계 분류:
    0: 정상 (Normal)       — 건강한 보행
    1: 전임상 (Pre-clinical) — AI만 감지 가능한 미세 변화
    2: 초기 (Early-stage)   — 경미하지만 명확한 초기 징후
    3: 임상 (Clinical)      — 확립된 질환 패턴 (파킨슨)

신뢰도 향상:
    - Data Augmentation   : 학습셋에만 노이즈/진폭 변조/시간이동 적용
    - Mixup 학습          : 인접 단계 경계 혼합으로 과적합 방지
    - Temperature Scaling : 예측 확신도 캘리브레이션 (ECE 측정)
"""

import argparse
import sys
import time
from pathlib import Path

# Windows 터미널 UTF-8 출력 강제
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
import yaml

from src.data.dataset import MultimodalGaitDataset
from src.data.synthetic_prodromal import generate_prodromal_dataset, CLASS_NAMES, STAGE_PROFILES
from src.models.multimodal_gait_net import MultimodalGaitNet
from src.utils.metrics import compute_metrics


# ── 색상 출력 (터미널) ─────────────────────────────────────────────────────────
def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"

BOLD    = lambda t: _c(t, "1")
GREEN   = lambda t: _c(t, "32")
CYAN    = lambda t: _c(t, "36")
YELLOW  = lambda t: _c(t, "33")
RED     = lambda t: _c(t, "31")
MAGENTA = lambda t: _c(t, "35")

STAGE_COLORS = [GREEN, CYAN, YELLOW, RED]
STAGE_RISK   = ["정상 · 위험 없음", "주의 · AI 감지 경계", "경고 · 초기 진단 권장", "위험 · 즉각 진료 필요"]


# ── Mixup ──────────────────────────────────────────────────────────────────────
def mixup_batch(batch: dict, alpha: float = 0.3):
    """배치 내 샘플을 무작위로 혼합해 결정 경계를 부드럽게 만든다."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(batch["imu"].size(0), device=batch["imu"].device)
    sensor_keys = [k for k in batch if k != "label"]
    mixed = {k: lam * batch[k] + (1 - lam) * batch[k][idx] for k in sensor_keys}
    return mixed, batch["label"], batch["label"][idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ── Temperature Scaling ────────────────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    """캘리브레이션용 온도 파라미터. Logits / T 로 확신도를 보정한다."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.1)

    def calibrate(self, model: nn.Module, val_loader: DataLoader, device: torch.device):
        """검증셋으로 NLL을 최소화하는 T를 찾는다."""
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch  = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("label")
                all_logits.append(model(batch).cpu())
                all_labels.append(labels.cpu())
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=200)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature.clamp(min=0.1), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())


# ── ECE (Expected Calibration Error) ──────────────────────────────────────────
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """예측 확신도와 실제 정확도의 편차를 측정한다. 낮을수록 캘리브레이션이 정확하다."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = accuracies[mask].mean()
        ece += mask.mean() * abs(avg_conf - avg_acc)
    return float(ece)


# ── 데이터 준비 ───────────────────────────────────────────────────────────────
def build_dataloaders(config: dict):
    data_cfg = config["data"]
    aug_cfg  = config.get("augmentation", {})

    raw = generate_prodromal_dataset(
        num_samples_per_stage=data_cfg.get("num_samples_per_stage", 100),
        num_frames=data_cfg["sequence_length"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
    )

    total   = len(raw["labels"])
    n_train = int(total * data_cfg["train_split"])
    n_val   = int(total * data_cfg["val_split"])
    n_test  = total - n_train - n_val

    g = torch.Generator().manual_seed(42)
    all_idx = torch.randperm(total, generator=g).tolist()
    train_idx = all_idx[:n_train]
    val_idx   = all_idx[n_train:n_train + n_val]
    test_idx  = all_idx[n_train + n_val:]

    common_kw = dict(
        sequence_length=data_cfg["sequence_length"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
    )

    full_aug  = MultimodalGaitDataset(raw, augment=True,  aug_cfg=aug_cfg, **common_kw)
    full_base = MultimodalGaitDataset(raw, augment=False,               **common_kw)

    train_ds = Subset(full_aug,  train_idx)
    val_ds   = Subset(full_base, val_idx)
    test_ds  = Subset(full_base, test_idx)

    bs = config["training"]["batch_size"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0),
        DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0),
        DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0),
        raw,
    )


# ── 1 epoch 학습 ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, mixup_alpha: float = 0.0):
    model.train()
    total_loss, preds, targets = 0.0, [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        if mixup_alpha > 0 and np.random.rand() < 0.5:
            mixed, y_a, y_b, lam = mixup_batch(batch, alpha=mixup_alpha)
            logits = model(mixed)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
            hard_labels = y_a if lam >= 0.5 else y_b
        else:
            labels = batch.pop("label")
            logits = model(batch)
            loss   = criterion(logits, labels)
            hard_labels = labels

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * hard_labels.size(0)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(hard_labels.cpu().numpy())

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    m = compute_metrics(targets, preds)
    m["loss"] = total_loss / len(preds)
    return m


@torch.no_grad()
def evaluate(model, loader, criterion, device, temperature: float = 1.0):
    model.eval()
    total_loss, preds, targets, all_probs = 0.0, [], [], []
    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")
        logits = model(batch) / max(temperature, 0.1)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        preds.append(probs.argmax(1))
        targets.append(labels.cpu().numpy())
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    probs   = np.concatenate(all_probs)
    m = compute_metrics(targets, preds)
    m["loss"] = total_loss / len(preds)
    m["ece"]  = compute_ece(probs, targets)
    return m


# ── 추론 데모 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference_demo(model, raw_data, config, device, temperature: float = 1.0):
    from src.data.preprocessing import preprocess_imu, preprocess_pressure, preprocess_magnetometer, preprocess_barometer

    data_cfg = config["data"]
    seq_len  = data_cfg["sequence_length"]
    grid     = tuple(data_cfg["pressure_grid_size"])

    model.eval()
    labels_arr = raw_data["labels"]

    print("\n" + "=" * 65)
    print(BOLD("  AI 이상징후 감지 추론 데모  [신발 전용 센서]"))
    print("=" * 65)

    rng = np.random.default_rng(99)

    for stage in range(4):
        idxs = np.where(labels_arr == stage)[0]
        idx  = rng.choice(idxs)

        imu  = torch.from_numpy(
            preprocess_imu(raw_data["imu"][idx], seq_len)
        ).unsqueeze(0).to(device)

        pres = torch.from_numpy(
            preprocess_pressure(raw_data["pressure"][idx], seq_len, grid)
        ).unsqueeze(0).to(device)

        mag  = preprocess_magnetometer(raw_data["magnetometer"][idx], seq_len)
        baro = preprocess_barometer(raw_data["barometer"][idx], seq_len)
        mb   = torch.from_numpy(np.concatenate([mag, baro], axis=0)).unsqueeze(0).to(device)

        logits = model({"imu": imu, "pressure": pres, "mag_baro": mb}) / max(temperature, 0.1)
        probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred   = int(probs.argmax())

        color     = STAGE_COLORS[pred]
        correct   = "✓" if pred == stage else "✗"
        truth     = CLASS_NAMES[stage]
        predicted = CLASS_NAMES[pred]

        print(f"\n  샘플 #{idx:03d}  |  실제: {BOLD(truth):12s}  |  예측: {color(BOLD(predicted))}  {correct}")
        print(f"  위험 등급: {color(STAGE_RISK[pred])}")
        print(f"  확률 분포:  (T={temperature:.2f})")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
            bar_len = int(prob * 30)
            bar     = "█" * bar_len + "░" * (30 - bar_len)
            marker  = " ◀" if i == pred else ""
            print(f"    {STAGE_COLORS[i](f'{name:6s}')}  {bar}  {prob*100:5.1f}%{marker}")

        p = STAGE_PROFILES[stage]
        print(f"\n  [센서 프로파일]  보행주파수 {p['freq']:.2f}Hz  |  떨림강도 {p['tremor_amp']:.3f}  |  비대칭 {int(p['asymmetry']*100)}%  |  팔흔들림 {int(p['arm_swing_scale']*100)}%")

    print("\n" + "=" * 65)


# ── 메인 학습 루프 ─────────────────────────────────────────────────────────────
def train(config: dict, output_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(BOLD(f"\n  장치: {device}"))

    train_loader, val_loader, test_loader, raw_data = build_dataloaders(config)
    print(f"  데이터셋  |  학습: {len(train_loader.dataset)}  검증: {len(val_loader.dataset)}  테스트: {len(test_loader.dataset)}")

    model = MultimodalGaitNet(config).to(device)
    print(f"  모델 파라미터: {model.get_num_trainable_params():,}\n")

    train_cfg    = config["training"]
    label_smooth = train_cfg.get("label_smoothing", 0.0)
    mixup_alpha  = train_cfg.get("mixup_alpha", 0.3)
    criterion    = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg["learning_rate"],
        epochs=train_cfg["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=train_cfg["scheduler"]["warmup_epochs"] / train_cfg["epochs"],
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1e3,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc  = 0.0
    patience_ctr  = 0
    es_cfg        = train_cfg["early_stopping"]
    history: dict = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_ece": []}

    print(BOLD(f"  학습 시작  ·  최대 {train_cfg['epochs']} 에폭  ·  LabelSmoothing={label_smooth}  ·  Mixup alpha={mixup_alpha}"))
    print("─" * 80)

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha)
        vl = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["val_acc"].append(vl["accuracy"])
        history["val_ece"].append(vl["ece"])

        improved = vl["accuracy"] > best_val_acc + es_cfg["min_delta"]
        if improved:
            best_val_acc = vl["accuracy"]
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc,
                "val_ece": vl["ece"],
                "config": config,
                "history": history,
            }, output_dir / "best_model.pt")
            marker = GREEN(" ★ best")
        else:
            patience_ctr += 1
            marker = f" ({patience_ctr}/{es_cfg['patience']})"

        print(
            f"  에폭 {epoch:3d}/{train_cfg['epochs']} | "
            f"학습 loss {tr['loss']:.4f}  acc {tr['accuracy']:.4f} | "
            f"검증 loss {vl['loss']:.4f}  acc {vl['accuracy']:.4f}  ECE {vl['ece']:.4f} | "
            f"{time.time()-t0:.1f}s{marker}"
        )

        if patience_ctr >= es_cfg["patience"]:
            print(YELLOW(f"\n  조기 종료 (에폭 {epoch}, patience={es_cfg['patience']})"))
            break

    # ── Temperature Calibration ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(BOLD("  Temperature Scaling 캘리브레이션"))
    print("=" * 80)

    ckpt = torch.load(output_dir / "best_model.pt", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    scaler = TemperatureScaler().to(device)
    T = scaler.calibrate(model, val_loader, device)
    print(f"  최적 온도 T = {GREEN(f'{T:.4f}')}")

    # 캘리브레이션 전/후 ECE 비교
    pre_ece  = evaluate(model, val_loader, criterion, device, temperature=1.0)["ece"]
    post_ece = evaluate(model, val_loader, criterion, device, temperature=T)["ece"]
    print(f"  ECE (before) : {YELLOW(f'{pre_ece:.4f}')}")
    print(f"  ECE (after)  : {GREEN(f'{post_ece:.4f}')}")

    # 체크포인트에 T 저장
    ckpt["temperature"] = T
    torch.save(ckpt, output_dir / "best_model.pt")

    # ── 최종 테스트 평가 ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(BOLD("  최종 테스트 평가"))
    print("=" * 80)

    test_m = evaluate(model, test_loader, criterion, device, temperature=T)

    acc_str = f"{test_m['accuracy']*100:.2f}%"
    print(f"  정확도 (Accuracy) : {GREEN(acc_str)}")
    print(f"  F1 (macro)        : {test_m['f1_macro']:.4f}")
    print(f"  정밀도 (Precision): {test_m['precision']:.4f}")
    print(f"  재현율 (Recall)   : {test_m['recall']:.4f}")
    test_ece_str = f"{test_m['ece']:.4f}"
    print(f"  ECE (calibrated)  : {GREEN(test_ece_str)}")

    print("\n  혼동 행렬 (Confusion Matrix):")
    cm = test_m["confusion_matrix"]
    header = "         " + "  ".join(f"{n:6s}" for n in CLASS_NAMES)
    print("  " + header)
    for i, row in enumerate(cm):
        row_str = "  ".join(
            GREEN(f"{v:6d}") if i == j else f"{v:6d}"
            for j, v in enumerate(row)
        )
        print(f"  {CLASS_NAMES[i]:6s} | {row_str}")

    print("\n  단계별 성능:")
    for i in range(4):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        print(f"  {STAGE_COLORS[i](f'{CLASS_NAMES[i]:6s}')}  정밀도 {prec:.3f}  재현율 {rec:.3f}  F1 {f1:.3f}")

    # ── 추론 데모 ─────────────────────────────────────────────────────────────
    run_inference_demo(model, raw_data, config, device, temperature=T)

    return test_m


def main():
    parser = argparse.ArgumentParser(description="질환 발병 전 이상징후 감지 모델 학습")
    parser.add_argument("--config",     default="configs/prodromal.yaml")
    parser.add_argument("--output-dir", default="outputs/prodromal")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train(config, Path(args.output_dir))


if __name__ == "__main__":
    main()
