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
from src.utils.model_manager import model_manager


def create_dataloaders(config: dict) -> tuple:
    """Create train/val/test dataloaders with synthetic data."""
    data_cfg = config["data"]

    dataset_dict = generate_synthetic_dataset(
        num_samples_per_class=50,
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


def train(config: dict, version: str = "1.0.0"):
    """Full training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Version: {version}")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"Dataset splits - Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Model
    model = MultimodalGaitNet(config).to(device)
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

    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    es_cfg = train_cfg["early_stopping"]

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print("-" * 70)

    for epoch in range(1, train_cfg["epochs"] + 1):
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
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= es_cfg["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (patience={es_cfg['patience']})")
            break

    # Final test evaluation
    print("\n" + "=" * 70)
    print("Final Test Evaluation")
    print("=" * 70)

    if best_model_state is None:
        best_model_state = model.state_dict()
    
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"Test Precision:  {test_metrics['precision']:.4f}")
    print(f"Test Recall:     {test_metrics['recall']:.4f}")

    # Register best model in ModelManager
    model_id = model_manager.save_model(
        model_state=best_model_state,
        config=config,
        metrics={
            "val_acc": float(best_val_acc),
            "test_acc": float(test_metrics["accuracy"]),
            "test_f1": float(test_metrics["f1_macro"])
        },
        version=version,
        model_type="basic",
        alias="latest"
    )
    print(f"\nModel registered with ID: {model_id}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train multimodal gait classifier")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--version", type=str, default="1.0.0",
        help="Model version (e.g. 1.0.0)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, version=args.version)


if __name__ == "__main__":
    main()
