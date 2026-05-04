"""Training pipeline for multimodal gait classification."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

from src.data.dataset import MultimodalGaitDataset
from src.data.synthetic import generate_synthetic_dataset
from src.models.multimodal_gait_net import MultimodalGaitNet
from src.utils.metrics import compute_metrics


def create_dataloaders(config: dict, num_samples: int = 50) -> tuple:
    """Create train/val/test dataloaders with synthetic data."""
    data_cfg = config["data"]

    dataset_dict = generate_synthetic_dataset(
        num_samples_per_class=num_samples,
        num_classes=data_cfg["num_classes"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )

    dataset = MultimodalGaitDataset(
        dataset_dict,
        sequence_length=data_cfg["sequence_length"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )

    total = len(dataset)
    train_n = int(total * data_cfg["train_split"])
    val_n = int(total * data_cfg["val_split"])
    test_n = total - train_n - val_n

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42),
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")

        logits = model(batch)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(all_preds)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")

        logits = model(batch)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(all_preds)
    return metrics


def _load_matching_checkpoint_weights(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """Load compatible checkpoint weights, skipping heads with changed shapes."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    current_state = model.state_dict()
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and value.size() == current_state[key].size()
    }
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(
        f"Loaded {len(filtered_state)} layers from {checkpoint_path}. "
        f"Missing: {len(missing)}, unexpected: {len(unexpected)}"
    )


def train(
    config: dict,
    output_dir: Path,
    num_samples: int = 50,
    resume: bool = False,
    checkpoint_path: Path | None = None,
):
    """Full training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(config, num_samples)
    print(f"Dataset splits - Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Model
    model = MultimodalGaitNet(config).to(device)
    if checkpoint_path and checkpoint_path.exists() and not resume:
        _load_matching_checkpoint_weights(model, checkpoint_path, device)
    print(f"Model parameters: {model.get_num_trainable_params():,}")

    # Training setup
    train_cfg = config["training"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler_cfg = train_cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"] - scheduler_cfg["warmup_epochs"],
    )

    # Resume from checkpoint
    start_epoch = 1
    best_val_acc = 0.0
    if resume:
        ckpt_path = output_dir / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_acc = ckpt["val_accuracy"]
            print(f"Resumed from epoch {ckpt['epoch']} (best val acc: {best_val_acc:.4f})")
        else:
            print(f"Warning: --resume 지정했지만 {ckpt_path} 없음. 처음부터 학습합니다.")

    # Training loop
    patience_counter = 0
    es_cfg = train_cfg["early_stopping"]
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print("-" * 70)

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        if epoch > scheduler_cfg["warmup_epochs"]:
            scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping / checkpointing
        if val_metrics["accuracy"] > best_val_acc + es_cfg["min_delta"]:
            best_val_acc = val_metrics["accuracy"]
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
            print(f"\nEarly stopping at epoch {epoch} (patience={es_cfg['patience']})")
            break

    # Final test evaluation
    print("\n" + "=" * 70)
    print("Final Test Evaluation")
    print("=" * 70)

    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"Test Precision:  {test_metrics['precision']:.4f}")
    print(f"Test Recall:     {test_metrics['recall']:.4f}")
    print(f"\nConfusion Matrix:\n{test_metrics['confusion_matrix']}")

    # Save final history (includes all epochs, not just up to best)
    torch.save({
        "epoch": checkpoint["epoch"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": checkpoint["optimizer_state_dict"],
        "val_accuracy": checkpoint["val_accuracy"],
        "config": config,
        "history": history,
    }, output_dir / "best_model.pt")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Shoealls — 멀티모달 보행 분류 모델 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="설정 파일 경로")
    parser.add_argument("--output-dir", default="outputs", help="체크포인트 저장 디렉터리")
    parser.add_argument("--samples", type=int, default=50,
                        help="클래스당 합성 샘플 수 (많을수록 정확도 향상)")
    parser.add_argument("--epochs", type=int, default=None, help="에포크 수 (설정 파일 오버라이드)")
    parser.add_argument("--lr", type=float, default=None, help="학습률 (설정 파일 오버라이드)")
    parser.add_argument("--resume", action="store_true", help="outputs/best_model.pt에서 재개")
    parser.add_argument("--checkpoint", default=None,
                        help="compatible pre-trained checkpoint for transfer learning")
    parser.add_argument("--verify", action="store_true",
                        help="학습 후 API 호환성 검증 실행")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI 오버라이드
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    test_metrics = train(config, output_dir, num_samples=args.samples,
                         resume=args.resume, checkpoint_path=checkpoint_path)

    # 학습 결과 JSON 저장
    import json
    result = {
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_f1_macro": round(test_metrics["f1_macro"], 4),
        "checkpoint": str(output_dir / "best_model.pt"),
        "config": args.config,
    }
    (output_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    print(f"\n체크포인트 저장 완료: {output_dir / 'best_model.pt'}")
    print(f"API에서 사용: --checkpoint_path {output_dir / 'best_model.pt'}")

    if args.verify:
        _verify_api_compat(config, output_dir / "best_model.pt")


def _verify_api_compat(config: dict, ckpt_path: Path):
    """저장된 체크포인트가 API와 호환되는지 빠른 검증."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    print("\n[검증] API 호환성 확인 중...")
    try:
        from api.schemas import SensorData
        from api.service import GaitMLService
        from api.examples import generate_sample_sensor_data

        svc = GaitMLService()
        sensor = SensorData(**generate_sample_sensor_data())
        result = svc.classify(sensor, ckpt=str(ckpt_path))
        print(f"  보행 분류 OK: {result.prediction_kr} ({result.confidence:.1%})")
        assert not result.is_demo_mode, "체크포인트 로드 실패"
        print("  API 호환성 검증 통과!")
    except Exception as e:
        print(f"  경고: 검증 실패 — {e}")


if __name__ == "__main__":
    main()
