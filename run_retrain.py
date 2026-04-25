"""실제 센서 데이터가 수집되면 사용하는 원커맨드 재학습 스크립트.

사용법:
  # 폴더 구조 데이터 (권장)
  python run_retrain.py --data-dir data/collected/ --strategy feature_extraction

  # .npz 파일
  python run_retrain.py --npz data/gait_data.npz --strategy partial

  # CSV 파일 패턴 (glob)
  python run_retrain.py \\
      --imu-files "data/imu_*.csv" \\
      --pressure-files "data/pressure_*.csv" \\
      --skeleton-files "data/skeleton_*.csv" \\
      --labels 0 0 1 1 2 2 3 3

전략:
  feature_extraction  백본 전체 동결 → head만 학습 (소규모 데이터 ≤50 샘플 권장)
  partial             마지막 N Transformer 레이어 + head 학습 (중규모 ≥100 샘플)
  lora                LoRA 어댑터 삽입 (대규모 데이터, VRAM 절약)
  full                전체 파라미터 학습 (충분한 데이터 ≥500 샘플)
"""

from __future__ import annotations
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models.multimodal_gait_net import MultimodalGaitNet
from src.data.dataset import MultimodalGaitDataset
from src.training.train import train_one_epoch, evaluate

# ── 상수 ────────────────────────────────────────────────────────────────────────

GAIT_CLASS_NAMES = ["normal", "antalgic", "ataxic", "parkinsonian"]
CONFIG_PATH = Path("configs/default.yaml")


# ── 데이터 로딩 ──────────────────────────────────────────────────────────────────

def load_dataset(args, config: dict) -> MultimodalGaitDataset:
    data_cfg = config["data"]
    seq_len   = data_cfg["sequence_length"]
    grid_size = tuple(data_cfg["pressure_grid_size"])
    n_joints  = data_cfg["skeleton_joints"]

    if args.npz:
        from src.data.adapters import NumpyDataAdapter
        print(f"[데이터] NPZ 파일 로드: {args.npz}")
        adapter = NumpyDataAdapter(args.npz)
        return adapter.to_dataset(seq_len, grid_size, n_joints)

    if args.data_dir:
        from src.data.adapters import FolderDataAdapter
        print(f"[데이터] 폴더 구조 로드: {args.data_dir}")
        adapter = FolderDataAdapter(
            args.data_dir,
            pressure_grid_size=grid_size,
            num_joints=n_joints,
            label_file=args.label_file,
        )
        return adapter.to_dataset(seq_len)

    if args.imu_files:
        from src.data.adapters import CSVDataAdapter
        imu_files = sorted(glob.glob(args.imu_files))
        pres_files = sorted(glob.glob(args.pressure_files))
        skel_files = sorted(glob.glob(args.skeleton_files))
        labels = [int(x) for x in args.labels]

        if not (len(imu_files) == len(pres_files) == len(skel_files) == len(labels)):
            raise ValueError(
                f"파일 수와 레이블 수가 일치하지 않습니다.\n"
                f"  IMU: {len(imu_files)}, Pressure: {len(pres_files)}, "
                f"Skeleton: {len(skel_files)}, Labels: {len(labels)}"
            )

        print(f"[데이터] CSV 파일 로드: {len(imu_files)}개 시행")
        adapter = CSVDataAdapter(
            imu_files, pres_files, skel_files, labels,
            pressure_grid_size=grid_size,
            num_joints=n_joints,
        )
        return adapter.to_dataset(seq_len)

    raise ValueError(
        "데이터 소스를 지정해야 합니다: --data-dir, --npz, 또는 --imu-files"
    )


def make_loaders(dataset: MultimodalGaitDataset, batch_size: int, val_ratio: float = 0.2):
    n = len(dataset)
    val_n   = max(1, int(n * val_ratio))
    train_n = n - val_n
    train_ds, val_ds = random_split(
        dataset, [train_n, val_n],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    print(f"[데이터] 학습: {train_n}샘플, 검증: {val_n}샘플 (batch_size={batch_size})")
    return train_loader, val_loader


# ── 전략 적용 ────────────────────────────────────────────────────────────────────

def apply_strategy(model: nn.Module, args) -> dict:
    strategy = args.strategy

    if strategy == "full":
        for p in model.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info = {"trainable": n, "total": total, "strategy": "full"}

    elif strategy == "feature_extraction":
        # 전체 동결 후 classifier만 열기
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info = {"trainable": n, "total": total, "strategy": "feature_extraction"}

    elif strategy == "partial":
        for p in model.parameters():
            p.requires_grad = False
        # classifier + fusion 열기
        for m in [model.classifier, model.fusion]:
            for p in m.parameters():
                p.requires_grad = True
        # 각 인코더 마지막 N 레이어 열기
        n_unfreeze = args.unfreeze_n
        for enc_name in ["imu_encoder", "pressure_encoder", "skeleton_encoder"]:
            enc = getattr(model, enc_name, None)
            if enc is None:
                continue
            # LSTM 기반 IMU 인코더: LSTM + projection 열기
            if hasattr(enc, "lstm"):
                for p in enc.lstm.parameters():
                    p.requires_grad = True
                if hasattr(enc, "proj"):
                    for p in enc.proj.parameters():
                        p.requires_grad = True
            # Conv 기반 인코더: 마지막 conv block 열기
            elif hasattr(enc, "convs"):
                convs = list(enc.convs)
                for block in convs[-n_unfreeze:]:
                    for p in block.parameters():
                        p.requires_grad = True
            # Fallback: 전체 열기
            else:
                for p in enc.parameters():
                    p.requires_grad = True
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info = {"trainable": n, "total": total, "strategy": "partial"}

    else:
        raise ValueError(f"알 수 없는 전략: {strategy}. 선택: feature_extraction, partial, lora, full")

    pct = info["trainable"] / info["total"] * 100
    print(f"[전략: {strategy}] 학습 파라미터: {info['trainable']:,} / {info['total']:,} ({pct:.1f}%)")
    return info


def make_optimizer(model: nn.Module, lr: float, strategy: str):
    if strategy == "partial":
        # 백본은 낮은 LR
        backbone_params, head_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(kw in name for kw in ["imu_encoder", "pressure_encoder", "skeleton_encoder"]):
                backbone_params.append(p)
            else:
                head_params.append(p)
        return torch.optim.AdamW([
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params,     "lr": lr},
        ], weight_decay=1e-4)
    else:
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4,
        )


# ── 기준선 평가 ───────────────────────────────────────────────────────────────────

def evaluate_baseline(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    criterion = nn.CrossEntropyLoss()
    return evaluate(model, loader, criterion, device)


# ── 출력 포맷 ────────────────────────────────────────────────────────────────────

def _bar(val: float, width: int = 20) -> str:
    n = round(val * width)
    return "█" * n + "░" * (width - n)


def print_comparison(before: dict, after: dict):
    metrics = ["accuracy", "f1_macro", "precision", "recall"]
    labels  = ["정확도", "F1(매크로)", "정밀도", "재현율"]
    print("\n" + "=" * 60)
    print("  재학습 전후 비교")
    print("=" * 60)
    print(f"  {'지표':<12}  {'재학습 전':>10}  {'재학습 후':>10}  {'변화':>8}")
    print("-" * 60)
    for key, label in zip(metrics, labels):
        b = before.get(key, 0.0)
        a = after.get(key, 0.0)
        delta = a - b
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<12}  {b:>10.4f}  {a:>10.4f}  {sign}{delta:>7.4f}")
    print("=" * 60)


def print_class_report(after: dict, num_classes: int):
    print("\n[클래스별 예측 분포]")
    preds = after.get("per_class_acc", {})
    for i in range(num_classes):
        name = GAIT_CLASS_NAMES[i] if i < len(GAIT_CLASS_NAMES) else str(i)
        acc  = preds.get(i, 0.0)
        print(f"  {name:<16} {_bar(acc)} {acc*100:5.1f}%")


# ── API 호환성 검증 ───────────────────────────────────────────────────────────────

def verify_api_compat(checkpoint_path: Path, config: dict) -> bool:
    print("\n[API 호환성 검증]")
    from fastapi.testclient import TestClient
    from api.main import app

    device = torch.device("cpu")
    cfg_data = config["data"]
    T = cfg_data["sequence_length"]
    H, W = cfg_data["pressure_grid_size"]
    J = cfg_data["skeleton_joints"]

    # 모델 로드 가능한지 확인
    try:
        model = MultimodalGaitNet(config).to(device)
        ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  ✓ 체크포인트 로드 성공: {checkpoint_path}")
    except (FileNotFoundError, KeyError, RuntimeError, torch.serialization.pickle.UnpicklingError) as e:
        print(f"  ✗ 체크포인트 로드 실패: {e}")
        return False

    # TestClient로 API 엔드포인트 호출
    try:
        client = TestClient(app, raise_server_exceptions=False)
        payload = {
            "imu":      [[0.1] * 6] * T,
            "pressure": [[0.5] * W for _ in range(H)],
            "skeleton": [[[0.0] * 3 for _ in range(J)]] * T,
        }
        resp = client.post("/api/classify", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert "gait_class" in data and "confidence" in data
            print(f"  ✓ /api/classify 응답: {data['gait_class']} ({data['confidence']:.3f})")
        else:
            print(f"  ✗ /api/classify 실패: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ API 호출 오류: {e}")
        return False

    print("  ✓ API 호환성 검증 통과")
    return True


# ── 메인 ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="실제 센서 데이터로 Shoealls 모델 재학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 데이터 소스 (셋 중 하나 필수)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--data-dir",  metavar="DIR",
                   help="FolderDataAdapter: subject별 폴더 루트 (subject_001/imu.csv ...)")
    g.add_argument("--npz",       metavar="FILE",
                   help="NumpyDataAdapter: .npz 파일 (imu, pressure, skeleton, labels 키 필요)")
    g.add_argument("--imu-files", metavar="GLOB",
                   help="CSVDataAdapter: IMU CSV glob (e.g. 'data/imu_*.csv')")

    # CSV 어댑터 보조 인수
    p.add_argument("--pressure-files", metavar="GLOB")
    p.add_argument("--skeleton-files",  metavar="GLOB")
    p.add_argument("--labels",  nargs="+", metavar="INT",
                   help="각 CSV 시행의 레이블 (0~3)")
    p.add_argument("--label-file", metavar="FILE",
                   help="FolderDataAdapter용 레이블 CSV (subject_id, label 열)")

    # 모델/학습
    p.add_argument("--strategy", default="feature_extraction",
                   choices=["feature_extraction", "partial", "lora", "full"],
                   help="파인튜닝 전략 (기본: feature_extraction)")
    p.add_argument("--base-checkpoint", metavar="FILE",
                   default="outputs/best_model.pt",
                   help="기반 체크포인트 (기본: outputs/best_model.pt)")
    p.add_argument("--output-dir", default="outputs/retrained",
                   help="재학습 결과 저장 디렉터리 (기본: outputs/retrained)")
    p.add_argument("--config", default=str(CONFIG_PATH),
                   help="학습 설정 YAML (기본: configs/default.yaml)")

    # 하이퍼파라미터
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--patience",     type=int,   default=8,
                   help="얼리 스토핑 patience (기본: 8)")
    p.add_argument("--unfreeze-n",   type=int,   default=1,
                   help="partial 전략: 마지막 N 레이어 해동 (기본: 1)")

    # 기타
    p.add_argument("--no-verify", action="store_true",
                   help="API 호환성 검증 건너뜀")

    args = p.parse_args()

    # 데이터 소스 미지정 시 오류
    if not args.data_dir and not args.npz and not args.imu_files:
        p.error("데이터 소스를 지정하세요: --data-dir, --npz, 또는 --imu-files")

    # CSV 어댑터 사용 시 보조 인수 확인
    if args.imu_files:
        if not args.pressure_files or not args.skeleton_files or not args.labels:
            p.error("--imu-files 사용 시 --pressure-files, --skeleton-files, --labels 필요")

    return args


def main():
    args = parse_args()

    # ── 설정 로드 ──────────────────────────────────────────────────────────────
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Shoealls 재학습 스크립트")
    print(f"{'='*60}")
    print(f"  전략     : {args.strategy}")
    print(f"  에포크   : {args.epochs}")
    print(f"  학습률   : {args.lr}")
    print(f"  배치     : {args.batch_size}")
    print(f"  출력     : {output_dir}")
    print(f"  디바이스 : {device}")
    print(f"{'='*60}\n")

    # ── 데이터 준비 ───────────────────────────────────────────────────────────
    dataset = load_dataset(args, config)
    print(f"  총 샘플: {len(dataset)}")

    if len(dataset) < 4:
        print("[경고] 샘플 수가 너무 적습니다 (최소 4개 필요). 데이터를 추가하세요.")
        sys.exit(1)

    train_loader, val_loader = make_loaders(dataset, args.batch_size)

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    base_ckpt = Path(args.base_checkpoint)
    model = MultimodalGaitNet(config).to(device)

    if base_ckpt.exists():
        ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[모델] 기반 체크포인트 로드: {base_ckpt}")
    else:
        print(f"[모델] 체크포인트 없음 → 랜덤 초기화 ({base_ckpt})")

    # ── 기준선 평가 (재학습 전) ───────────────────────────────────────────────
    print("\n[기준선 평가] 재학습 전 검증 세트 성능...")
    criterion = nn.CrossEntropyLoss()
    before_metrics = evaluate(model, val_loader, criterion, device)
    print(f"  정확도: {before_metrics['accuracy']:.4f}  "
          f"F1: {before_metrics.get('f1_macro', 0.0):.4f}  "
          f"손실: {before_metrics['loss']:.4f}")

    # ── 전략 적용 ──────────────────────────────────────────────────────────────
    strategy_info = apply_strategy(model, args)

    # ── 옵티마이저 / 스케줄러 ─────────────────────────────────────────────────
    optimizer = make_optimizer(model, args.lr, args.strategy)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── 학습 루프 ──────────────────────────────────────────────────────────────
    print(f"\n[학습 시작] {args.epochs} 에포크")
    print("-" * 60)

    best_val_acc   = before_metrics["accuracy"]
    patience_count = 0
    history        = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_ckpt_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(train_m["accuracy"])
        history["val_acc"].append(val_m["accuracy"])

        elapsed = time.time() - t0
        improved = "✓" if val_m["accuracy"] > best_val_acc else " "
        print(
            f"  {improved} Epoch {epoch:3d}/{args.epochs} | "
            f"Train {train_m['loss']:.4f}/{train_m['accuracy']:.4f} | "
            f"Val {val_m['loss']:.4f}/{val_m['accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy":         best_val_acc,
                "strategy":             strategy_info,
                "config":               config,
            }, best_ckpt_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n  얼리 스토핑 (patience={args.patience}, epoch={epoch})")
                break

    # ── 최종 평가 (재학습 후) ─────────────────────────────────────────────────
    print(f"\n[최종 평가] 최적 체크포인트 로드: {best_ckpt_path}")
    best_state = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    after_metrics = evaluate(model, val_loader, criterion, device)

    # ── 결과 비교 ──────────────────────────────────────────────────────────────
    print_comparison(before_metrics, after_metrics)

    num_classes = config["data"]["num_classes"]
    if "per_class_acc" in after_metrics:
        print_class_report(after_metrics, num_classes)

    # 히스토리 저장
    np.save(output_dir / "history.npy", history)
    print(f"\n  체크포인트  → {best_ckpt_path}")
    print(f"  학습 이력   → {output_dir / 'history.npy'}")

    # ── API 호환성 검증 ───────────────────────────────────────────────────────
    if not args.no_verify:
        ok = verify_api_compat(best_ckpt_path, config)
        if not ok:
            print("\n[경고] API 호환성 검증 실패. 체크포인트를 확인하세요.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  재학습 완료  정확도: {before_metrics['accuracy']:.4f} → {after_metrics['accuracy']:.4f}")
    print(f"  최적 에포크 저장 위치: {best_ckpt_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
