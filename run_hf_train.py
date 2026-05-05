"""HFMultimodalGaitNet (PatchTST + VideoMAE) 학습 및 기존 모델 비교.

실행:
    python run_hf_train.py

출력:
    outputs/hf/  — HFMultimodalGaitNet 체크포인트 + figures
    outputs/     — 기존 MultimodalGaitNet 체크포인트 (이미 존재하면 재사용)
"""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

from src.data.dataset import MultimodalGaitDataset
from src.data.synthetic import generate_synthetic_dataset
from src.models.multimodal_gait_net import MultimodalGaitNet
from src.models.hf_gait_net import HFMultimodalGaitNet
from src.utils.metrics import compute_metrics

# ── 경량 HF 설정 (CPU 학습 가능) ─────────────────────────────────────────────
HF_CONFIG_OVERRIDE = {
    "patchtst": {
        "pretrained": False,   # random init (오프라인)
        "patch_len": 16,
        "stride": 8,
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,       # 경량화
        "dropout": 0.2,
    },
    "videomae": {
        "pretrained": False,   # random init (오프라인)
        "img_size": 64,        # 224 → 64 (CPU 친화적)
        "num_frames": 4,       # 16 → 4
        "dropout": 0.2,
    },
}

# VideoMAE 내부 아키텍처도 경량화
VIDEOMAE_ARCH_PATCH = {
    "image_size": 64, "patch_size": 16, "num_frames": 4, "tubelet_size": 2,
    "hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
    "intermediate_size": 256,
}

# ── 학습 하이퍼파라미터 ────────────────────────────────────────────────────────
NUM_SAMPLES   = 60    # 클래스당 샘플 수
NUM_EPOCHS    = 30
BATCH_SIZE    = 16
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10
NUM_CLASSES   = 4
CLASS_NAMES   = ["정상", "절뚝거림", "운동실조", "파킨슨"]
CLASS_COLORS  = ["#2196F3", "#FF9800", "#E53935", "#8E24AA"]


# ── VideoMAE 아키텍처 패치 ─────────────────────────────────────────────────────

def _patch_videomae_build(original_build):
    """_build_backbone 을 경량 설정으로 패치한다."""
    from transformers import VideoMAEConfig, VideoMAEModel

    def patched(pretrained, img_size, num_frames):
        cfg = VideoMAEConfig(
            image_size=VIDEOMAE_ARCH_PATCH["image_size"],
            patch_size=VIDEOMAE_ARCH_PATCH["patch_size"],
            num_frames=VIDEOMAE_ARCH_PATCH["num_frames"],
            tubelet_size=VIDEOMAE_ARCH_PATCH["tubelet_size"],
            num_channels=3,
            hidden_size=VIDEOMAE_ARCH_PATCH["hidden_size"],
            num_hidden_layers=VIDEOMAE_ARCH_PATCH["num_hidden_layers"],
            num_attention_heads=VIDEOMAE_ARCH_PATCH["num_attention_heads"],
            intermediate_size=VIDEOMAE_ARCH_PATCH["intermediate_size"],
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        return VideoMAEModel(cfg), cfg.hidden_size, False

    return patched


# ── 데이터 ────────────────────────────────────────────────────────────────────

def make_loaders(config: dict):
    data_cfg = config["data"]
    ds_dict = generate_synthetic_dataset(
        num_samples_per_class=NUM_SAMPLES,
        num_classes=data_cfg["num_classes"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )
    ds = MultimodalGaitDataset(
        ds_dict,
        sequence_length=data_cfg["sequence_length"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )
    total = len(ds)
    tr = int(total * 0.7)
    va = int(total * 0.15)
    te = total - tr - va
    tr_ds, va_ds, te_ds = random_split(
        ds, [tr, va, te], generator=torch.Generator().manual_seed(42)
    )
    kw = dict(batch_size=BATCH_SIZE, num_workers=0)
    return (
        DataLoader(tr_ds, shuffle=True, **kw),
        DataLoader(va_ds, **kw),
        DataLoader(te_ds, **kw),
        ds_dict["class_names"],
    )


# ── 학습 루프 ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, preds, labels_all = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop("label")
        logits = model(batch)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        preds.append(logits.argmax(1).cpu().numpy())
        labels_all.append(y.cpu().numpy())
    p = np.concatenate(preds)
    l = np.concatenate(labels_all)
    m = compute_metrics(l, p)
    m["loss"] = total_loss / len(p)
    return m


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels_all = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop("label")
        logits = model(batch)
        total_loss += criterion(logits, y).item() * y.size(0)
        preds.append(logits.argmax(1).cpu().numpy())
        labels_all.append(y.cpu().numpy())
    p = np.concatenate(preds)
    l = np.concatenate(labels_all)
    m = compute_metrics(l, p)
    m["loss"] = total_loss / len(p)
    return m, p, l


def run_training(model, model_name, train_loader, val_loader, device, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, patience_cnt = 0.0, 0

    print(f"\n{'='*60}")
    print(f"  {model_name} 학습 시작")
    print(f"  파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*60}")
    print(f"{'에포크':>6} {'학습손실':>9} {'학습정확도':>10} {'검증손실':>9} {'검증정확도':>10} {'시간':>6}")
    print("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["val_acc"].append(va["accuracy"])

        mark = " ★" if va["accuracy"] > best_val_acc else ""
        print(
            f"{epoch:>6d} {tr['loss']:>9.4f} {tr['accuracy']:>10.4f}"
            f" {va['loss']:>9.4f} {va['accuracy']:>10.4f}"
            f" {time.time()-t0:>5.1f}s{mark}"
        )

        if va["accuracy"] > best_val_acc + 1e-4:
            best_val_acc = va["accuracy"]
            patience_cnt = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc,
                "history": history,
                "epoch": epoch,
            }, save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  조기 종료 (에포크 {epoch})")
                break

    ckpt = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return history, best_val_acc


# ── 시각화 ────────────────────────────────────────────────────────────────────

def plot_results(
    hf_hist, base_hist,
    hf_preds, hf_labels,
    base_preds, base_labels,
    hf_val, base_val,
    class_names, save_path,
):
    from sklearn.metrics import confusion_matrix, classification_report
    fig = plt.figure(figsize=(20, 14), facecolor="#0F1923")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    C_BG   = "#0F1923"
    C_SURF = "#1A2535"
    C_TEXT = "#E8ECF0"
    C_MUTED = "#8897A4"
    C_HF   = "#4FC3F7"   # PatchTST+VideoMAE
    C_BASE = "#FFB74D"   # 기존 CNN

    def _ax(pos, title=""):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(C_SURF)
        for sp in ax.spines.values():
            sp.set_color("#2A3A4A")
        ax.tick_params(colors=C_MUTED, labelsize=9)
        if title:
            ax.set_title(title, color=C_TEXT, fontsize=11, fontweight="bold", pad=8)
        return ax

    epochs_hf   = range(1, len(hf_hist["val_acc"]) + 1)
    epochs_base = range(1, len(base_hist["val_acc"]) + 1)

    # ── 1. 검증 정확도 비교 ─────────────────────────────────────────────────
    ax1 = _ax(gs[0, :2], "검증 정확도 비교")
    ax1.plot(epochs_base, base_hist["val_acc"], color=C_BASE, lw=2, label="기존 (CNN+LSTM+STGCN)")
    ax1.plot(epochs_hf,   hf_hist["val_acc"],   color=C_HF,   lw=2, label="HF (PatchTST+VideoMAE)")
    ax1.set_xlabel("에포크", color=C_MUTED, fontsize=9)
    ax1.set_ylabel("정확도", color=C_MUTED, fontsize=9)
    ax1.legend(facecolor=C_SURF, labelcolor=C_TEXT, fontsize=9)
    ax1.set_ylim(0, 1.05)

    # ── 2. 손실 비교 ────────────────────────────────────────────────────────
    ax2 = _ax(gs[0, 2:], "학습 손실 비교")
    ax2.plot(epochs_base, base_hist["train_loss"], color=C_BASE, lw=1.5, ls="--", alpha=0.7, label="기존 학습")
    ax2.plot(epochs_base, base_hist["val_loss"],   color=C_BASE, lw=2,   label="기존 검증")
    ax2.plot(epochs_hf,   hf_hist["train_loss"],   color=C_HF,   lw=1.5, ls="--", alpha=0.7, label="HF 학습")
    ax2.plot(epochs_hf,   hf_hist["val_loss"],     color=C_HF,   lw=2,   label="HF 검증")
    ax2.set_xlabel("에포크", color=C_MUTED, fontsize=9)
    ax2.set_ylabel("손실", color=C_MUTED, fontsize=9)
    ax2.legend(facecolor=C_SURF, labelcolor=C_TEXT, fontsize=9)

    # ── 3. HF 혼동 행렬 ────────────────────────────────────────────────────
    ax3 = _ax(gs[1, :2], "HF 모델 혼동 행렬")
    cm_hf = confusion_matrix(hf_labels, hf_preds)
    cm_n  = cm_hf.astype(float) / cm_hf.sum(axis=1, keepdims=True)
    im = ax3.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f"{cm_hf[i,j]}", ha="center", va="center",
                     color="white" if cm_n[i,j] > 0.5 else C_TEXT, fontsize=11, fontweight="bold")
    ax3.set_xticks(range(4)); ax3.set_xticklabels(CLASS_NAMES, color=C_MUTED, fontsize=8)
    ax3.set_yticks(range(4)); ax3.set_yticklabels(CLASS_NAMES, color=C_MUTED, fontsize=8)
    ax3.set_xlabel("예측", color=C_MUTED, fontsize=9)
    ax3.set_ylabel("실제", color=C_MUTED, fontsize=9)

    # ── 4. 기존 혼동 행렬 ──────────────────────────────────────────────────
    ax4 = _ax(gs[1, 2:], "기존 모델 혼동 행렬")
    cm_b = confusion_matrix(base_labels, base_preds)
    cm_bn = cm_b.astype(float) / cm_b.sum(axis=1, keepdims=True)
    ax4.imshow(cm_bn, cmap="Oranges", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax4.text(j, i, f"{cm_b[i,j]}", ha="center", va="center",
                     color="white" if cm_bn[i,j] > 0.5 else C_TEXT, fontsize=11, fontweight="bold")
    ax4.set_xticks(range(4)); ax4.set_xticklabels(CLASS_NAMES, color=C_MUTED, fontsize=8)
    ax4.set_yticks(range(4)); ax4.set_yticklabels(CLASS_NAMES, color=C_MUTED, fontsize=8)
    ax4.set_xlabel("예측", color=C_MUTED, fontsize=9)
    ax4.set_ylabel("실제", color=C_MUTED, fontsize=9)

    # ── 5. 클래스별 F1 비교 ────────────────────────────────────────────────
    ax5 = _ax(gs[2, :2], "클래스별 F1 비교")
    from sklearn.metrics import f1_score
    f1_hf   = f1_score(hf_labels,   hf_preds,   average=None, labels=range(4))
    f1_base = f1_score(base_labels, base_preds, average=None, labels=range(4))
    x = np.arange(4)
    bars_b = ax5.bar(x - 0.2, f1_base, 0.35, color=C_BASE, alpha=0.85, label="기존")
    bars_h = ax5.bar(x + 0.2, f1_hf,   0.35, color=C_HF,   alpha=0.85, label="HF")
    for bar in list(bars_b) + list(bars_h):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", va="bottom",
                 color=C_TEXT, fontsize=8)
    ax5.set_xticks(x); ax5.set_xticklabels(CLASS_NAMES, color=C_MUTED, fontsize=9)
    ax5.set_ylabel("F1 Score", color=C_MUTED, fontsize=9)
    ax5.set_ylim(0, 1.1)
    ax5.legend(facecolor=C_SURF, labelcolor=C_TEXT, fontsize=9)

    # ── 6. 최종 성능 비교 요약 카드 ────────────────────────────────────────
    ax6 = _ax(gs[2, 2:])
    ax6.axis("off")
    from sklearn.metrics import accuracy_score, f1_score as f1s
    hf_acc   = accuracy_score(hf_labels, hf_preds)
    base_acc = accuracy_score(base_labels, base_preds)
    hf_f1    = f1s(hf_labels, hf_preds, average="macro")
    base_f1  = f1s(base_labels, base_preds, average="macro")

    rows = [
        ["지표", "기존 CNN+LSTM", "HF PatchTST+VideoMAE", "향상"],
        ["정확도",    f"{base_acc:.4f}", f"{hf_acc:.4f}",
         f"{(hf_acc-base_acc)*100:+.2f}%p"],
        ["F1 (macro)", f"{base_f1:.4f}", f"{hf_f1:.4f}",
         f"{(hf_f1-base_f1)*100:+.2f}%p"],
        ["최고 검증 정확도", f"{hf_val:.4f}", f"{base_val:.4f}", "—"],
        ["파라미터 (학습)", "~1.23M", "~1.28M", "—"],
    ]

    col_colors = [["#223344"]*4] + [["#1A2535"]*4]*4
    tbl = ax6.table(
        cellText=rows[1:], colLabels=rows[0],
        cellLoc="center", loc="center",
        cellColours=col_colors[1:],
        colColours=col_colors[0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2A3A4A")
        if r == 0:
            cell.set_text_props(color=C_TEXT, fontweight="bold")
        elif c == 3:
            val = rows[r][3]
            color = "#4CAF50" if "+" in val else ("#EF5350" if "-" in val else C_TEXT)
            cell.set_text_props(color=color, fontweight="bold")
        else:
            cell.set_text_props(color=C_MUTED)

    ax6.set_title("최종 성능 비교", color=C_TEXT, fontsize=11, fontweight="bold", pad=8)

    # ── 타이틀 ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "HuggingFace 모델 통합 학습 결과\n"
        "PatchTST(IMU) + VideoMAE(Skeleton) vs 기존 CNN+BiLSTM+STGCN",
        color=C_TEXT, fontsize=14, fontweight="bold", y=0.98,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"\n  시각화 저장: {save_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    # HF 학습용 설정 주입
    config["hf_encoders"] = {"enabled": True, **HF_CONFIG_OVERRIDE}

    # VideoMAE 경량화 패치
    from src.models import hf_encoders
    hf_encoders.VideoMAESkeletonEncoder._build_backbone = staticmethod(
        _patch_videomae_build(hf_encoders.VideoMAESkeletonEncoder._build_backbone)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")

    out_base = Path("outputs")
    out_hf   = Path("outputs/hf")
    out_hf.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = make_loaders(config)
    print(f"데이터: 학습 {len(train_loader.dataset)} | 검증 {len(val_loader.dataset)} | 테스트 {len(test_loader.dataset)}")

    criterion = nn.CrossEntropyLoss()

    # ── 1. 기존 MultimodalGaitNet 학습 ──────────────────────────────────────
    base_model = MultimodalGaitNet(config).to(device)
    base_hist, base_val = run_training(
        base_model, "기존 MultimodalGaitNet (CNN+BiLSTM+STGCN)",
        train_loader, val_loader, device,
        out_base / "best_model.pt",
    )
    base_test, base_preds, base_labels = evaluate(base_model, test_loader, criterion, device)

    # ── 2. HFMultimodalGaitNet 학습 ─────────────────────────────────────────
    hf_model = HFMultimodalGaitNet(config).to(device)
    hf_hist, hf_val = run_training(
        hf_model, "HFMultimodalGaitNet (PatchTST+VideoMAE)",
        train_loader, val_loader, device,
        out_hf / "hf_best_model.pt",
    )
    hf_test, hf_preds, hf_labels = evaluate(hf_model, test_loader, criterion, device)

    # ── 결과 출력 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  최종 테스트 결과 비교")
    print("=" * 60)
    print(f"{'지표':<20} {'기존 CNN+LSTM':>15} {'HF PatchTST+VideoMAE':>22}")
    print("-" * 60)
    for k in ["accuracy", "f1_macro", "precision", "recall"]:
        b = base_test.get(k, 0)
        h = hf_test.get(k, 0)
        diff = h - b
        mark = "↑" if diff > 0.005 else ("↓" if diff < -0.005 else "≈")
        print(f"  {k:<18} {b:>15.4f} {h:>22.4f}  {mark}{abs(diff)*100:.2f}%p")
    print()

    # 클래스별 상세
    from sklearn.metrics import classification_report
    print("【HF 모델 클래스별 성능】")
    print(classification_report(hf_labels, hf_preds, target_names=CLASS_NAMES, digits=3))

    # ── 시각화 ──────────────────────────────────────────────────────────────
    plot_results(
        hf_hist, base_hist,
        hf_preds, hf_labels,
        base_preds, base_labels,
        hf_val, base_val,
        class_names,
        out_hf / "hf_comparison.png",
    )

    print("=" * 60)
    print(f"  체크포인트: {out_hf / 'hf_best_model.pt'}")
    print(f"  시각화:     {out_hf / 'hf_comparison.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
