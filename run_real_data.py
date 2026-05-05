"""실제 데이터 수집 후 알고리즘 실행 스크립트.

=== 사용법 ===

1. 폴더 구조 데이터:
    python run_real_data.py --data-dir data/collected/ --format folder

2. NPZ 파일:
    python run_real_data.py --data-dir data/gait_data.npz --format npz

3. 결과물:
    outputs/real/
    ├── best_model.pt            # 학습된 모델
    ├── figures/                  # 영문 차트
    └── report/                   # 한글 보고서 (3페이지)
        ├── report_p1_summary.png
        ├── report_p2_detail.png
        └── report_p3_ablation.png
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

from src.data.adapters import FolderDataAdapter, NumpyDataAdapter
from src.models.multimodal_gait_net import MultimodalGaitNet
from src.training.train import train_one_epoch, evaluate
from src.utils.metrics import compute_metrics
from src.validation.report import generate_report


def load_dataset(args, config):
    """데이터 형식에 맞는 어댑터로 데이터셋 로드."""
    data_cfg = config["data"]
    seq_len = data_cfg["sequence_length"]
    grid_size = tuple(data_cfg["pressure_grid_size"])
    num_joints = data_cfg["skeleton_joints"]

    if args.format == "folder":
        adapter = FolderDataAdapter(
            data_root=args.data_dir,
            pressure_grid_size=grid_size,
            num_joints=num_joints,
            has_header=not args.no_header,
            has_timestamp_col=args.has_timestamp,
            label_file=args.label_file,
        )
        return adapter.to_dataset(sequence_length=seq_len)

    elif args.format == "npz":
        adapter = NumpyDataAdapter(args.data_dir)
        return adapter.to_dataset(
            sequence_length=seq_len, grid_size=grid_size, num_joints=num_joints
        )

    else:
        raise ValueError(f"Unsupported format: {args.format}")


def split_dataset(dataset, config):
    """Train/Val/Test 분할."""
    data_cfg = config["data"]
    total = len(dataset)
    train_n = int(total * data_cfg["train_split"])
    val_n = int(total * data_cfg["val_split"])
    test_n = total - train_n - val_n

    return random_split(
        dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42),
    )


def run_ablation(model, dataset, device, config):
    """모달리티 기여도 분석 (Ablation Study)."""
    loader = DataLoader(dataset, batch_size=32)

    strategies = {
        "IMU only": {"pressure": True, "skeleton": True},
        "Pressure only": {"imu": True, "skeleton": True},
        "Skeleton only": {"imu": True, "pressure": True},
        "IMU + Pressure": {"skeleton": True},
        "IMU + Skeleton": {"pressure": True},
        "Pressure + Skeleton": {"imu": True},
        "All (Fusion)": {},
    }

    results = {}
    model.eval()
    for name, mask in strategies.items():
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("label")
                for key in mask:
                    batch[key] = torch.zeros_like(batch[key])
                logits = model(batch)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        acc = (preds == labels).mean()
        results[name] = acc
        print(f"  {name:25s}: {acc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="실제 데이터로 멀티모달 보행 AI 알고리즘 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 폴더 구조 데이터
  python run_real_data.py --data-dir data/collected/ --format folder

  # NPZ 파일
  python run_real_data.py --data-dir data/gait_data.npz --format npz

  # 옵션 조정
  python run_real_data.py --data-dir data/ --format folder \\
      --label-file labels.csv --epochs 50 --lr 0.0005
        """,
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="데이터 경로 (폴더 또는 .npz 파일)")
    parser.add_argument("--format", type=str, choices=["folder", "npz"],
                        default="folder", help="데이터 형식")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="설정 파일 경로")
    parser.add_argument("--output-dir", type=str, default="outputs/real",
                        help="결과 저장 경로")
    parser.add_argument("--label-file", type=str, default="labels.csv",
                        help="라벨 파일명 (폴더 모드)")
    parser.add_argument("--no-header", action="store_true",
                        help="CSV 파일에 헤더가 없는 경우")
    parser.add_argument("--has-timestamp", action="store_true",
                        help="CSV 첫 열이 타임스탬프인 경우")
    parser.add_argument("--epochs", type=int, default=None,
                        help="학습 에포크 수 (기본: config 값)")
    parser.add_argument("--lr", type=float, default=None,
                        help="학습률 (기본: config 값)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="배치 크기 (기본: config 값)")
    args = parser.parse_args()

    # Config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── STEP 1: 데이터 로드 ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: 데이터 로드")
    print("=" * 60)
    dataset = load_dataset(args, config)
    print(f"총 샘플 수: {len(dataset)}")

    # ── STEP 2: 데이터 분할 ──────────────────────────────────────────
    train_ds, val_ds, test_ds = split_dataset(dataset, config)
    print(f"분할: 학습 {len(train_ds)} / 검증 {len(val_ds)} / 테스트 {len(test_ds)}")

    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # ── STEP 3: 모델 학습 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: 모델 학습")
    print("=" * 60)

    model = MultimodalGaitNet(config).to(device)
    print(f"모델 파라미터: {model.get_num_trainable_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"] - train_cfg["scheduler"]["warmup_epochs"],
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    es_cfg = train_cfg["early_stopping"]

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device)

        if epoch > train_cfg["scheduler"]["warmup_epochs"]:
            scheduler.step()

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(train_m["accuracy"])
        history["val_acc"].append(val_m["accuracy"])

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
              f"Train Loss: {train_m['loss']:.4f} Acc: {train_m['accuracy']:.4f} | "
              f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.4f} | "
              f"{elapsed:.1f}s")

        if val_m["accuracy"] > best_val_acc + es_cfg["min_delta"]:
            best_val_acc = val_m["accuracy"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "config": config,
                "history": history,
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= es_cfg["patience"]:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save final history
    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=True)
    checkpoint["history"] = history
    torch.save(checkpoint, output_dir / "best_model.pt")

    # ── STEP 4: 테스트 평가 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: 테스트 평가")
    print("=" * 60)

    model.load_state_dict(checkpoint["model_state_dict"])
    test_m = evaluate(model, test_loader, criterion, device)
    print(f"테스트 정확도:  {test_m['accuracy']:.4f}")
    print(f"테스트 F1:     {test_m['f1_macro']:.4f}")

    # Collect predictions for visualization
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    # ── STEP 5: Ablation Study ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: 모달리티 기여도 분석")
    print("=" * 60)
    ablation_results = run_ablation(model, test_ds, device, config)

    # ── STEP 6: 보고서 생성 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: 보고서 생성")
    print("=" * 60)

    class_names = config["data"].get("class_names", [
        "normal", "antalgic", "ataxic", "parkinsonian"
    ])

    generate_report(
        history=history,
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        class_names=class_names,
        ablation_results=ablation_results,
        model_params=model.get_num_trainable_params(),
        save_dir=output_dir / "report",
    )

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"결과 위치: {output_dir}/")
    print(f"  모델:   {output_dir}/best_model.pt")
    print(f"  보고서: {output_dir}/report/")


if __name__ == "__main__":
    main()
