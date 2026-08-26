"""Train HAR baselines on PAMAP2/WISDM/UCI-HAR or ShoeAlls local windows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.har_dataset import (
    load_har_npz,
    load_hf_har_dataset,
    split_dataset,
    synthetic_har_dataset,
)
from src.models.har_transformer import PatchTSTClassifier, freeze_encoder
from src.utils.metrics import compute_metrics


def build_dataset(config: dict):
    data_cfg = config["data"]
    source = data_cfg["source"]
    if source == "hf":
        return load_hf_har_dataset(
            repo_id=data_cfg["hf_repo"],
            split=data_cfg.get("hf_split", "train"),
            sequence_length=data_cfg["sequence_length"],
            max_samples=data_cfg.get("max_samples"),
        )
    if source == "npz":
        return load_har_npz(
            data_cfg["local_npz"],
            sequence_length=data_cfg["sequence_length"],
        )
    if source == "synthetic":
        return synthetic_har_dataset(
            num_samples=data_cfg.get("max_samples") or 128,
            channels=data_cfg["channels"],
            sequence_length=data_cfg["sequence_length"],
            num_classes=data_cfg["num_classes"],
            seed=data_cfg.get("seed", 42),
        )
    raise ValueError(f"unknown data source: {source}")


def infer_dataset_shape(dataset) -> tuple[int, int]:
    sample = dataset[0]["sensor"]
    return int(sample.shape[0]), int(sample.shape[1])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    preds, labels = [], []
    for batch in loader:
        x = batch["sensor"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    return summarize_epoch(total_loss, preds, labels)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    for batch in loader:
        x = batch["sensor"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return summarize_epoch(total_loss, preds, labels)


def summarize_epoch(total_loss, preds, labels):
    preds_arr = np.concatenate(preds)
    labels_arr = np.concatenate(labels)
    metrics = compute_metrics(labels_arr, preds_arr)
    metrics["loss"] = total_loss / len(labels_arr)
    return metrics


def build_model(config: dict, channels: int, num_classes: int) -> PatchTSTClassifier:
    model_cfg = config["model"]
    return PatchTSTClassifier(
        in_channels=channels,
        num_classes=num_classes,
        sequence_length=config["data"]["sequence_length"],
        patch_size=model_cfg["patch_size"],
        stride=model_cfg["stride"],
        embed_dim=model_cfg["embed_dim"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        ff_dim=model_cfg["ff_dim"],
        dropout=model_cfg["dropout"],
    )


def load_checkpoint(model, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    current = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)} compatible tensors from {checkpoint_path}")


def train(config: dict, output_dir: Path, checkpoint: Path | None = None, freeze: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config)
    channels, sequence_length = infer_dataset_shape(dataset)
    labels = [int(dataset[idx]["label"]) for idx in range(len(dataset))]
    num_classes = max(labels) + 1

    config["data"]["channels"] = channels
    config["data"]["sequence_length"] = sequence_length
    config["data"]["num_classes"] = num_classes

    train_ds, val_ds, test_ds = split_dataset(
        dataset,
        config["data"]["train_split"],
        config["data"]["val_split"],
        seed=config["data"].get("seed", 42),
    )
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = build_model(config, channels, num_classes).to(device)
    if checkpoint:
        load_checkpoint(model, checkpoint, device)
    if freeze:
        freeze_encoder(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    es_cfg = config["training"]["early_stopping"]
    best_val_acc = -1.0
    patience = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset: {len(dataset)} windows, channels={channels}, length={sequence_length}, classes={num_classes}")
    print(f"Model trainable params: {model.get_num_trainable_params():,}")

    for epoch in range(1, config["training"]["epochs"] + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"{time.time() - start:.1f}s"
        )

        if val_metrics["accuracy"] > best_val_acc + es_cfg["min_delta"]:
            best_val_acc = val_metrics["accuracy"]
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "history": history,
                    "val_accuracy": best_val_acc,
                    "epoch": epoch,
                },
                output_dir / "best_har_model.pt",
            )
        else:
            patience += 1
            if patience >= es_cfg["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint_data = torch.load(output_dir / "best_har_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    result = {
        "phase": config.get("experiment", {}).get("phase"),
        "test_accuracy": round(float(test_metrics["accuracy"]), 4),
        "test_f1_macro": round(float(test_metrics["f1_macro"]), 4),
        "checkpoint": str(output_dir / "best_har_model.pt"),
        "config": config,
    }
    (output_dir / "har_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Test accuracy={test_metrics['accuracy']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")
    print(f"Saved checkpoint: {output_dir / 'best_har_model.pt'}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train ShoeAlls HAR/IMU baselines")
    parser.add_argument("--config", default="configs/har_baseline.yaml")
    parser.add_argument("--source", choices=["synthetic", "hf", "npz"], default=None)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--local-npz", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.source:
        config["data"]["source"] = args.source
    if args.hf_repo:
        config["data"]["hf_repo"] = args.hf_repo
    if args.local_npz:
        config["data"]["local_npz"] = args.local_npz
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.max_samples:
        config["data"]["max_samples"] = args.max_samples

    output_dir = Path(args.output_dir or config["experiment"]["output_dir"])
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    train(config, output_dir, checkpoint=checkpoint, freeze=args.freeze_encoder)


if __name__ == "__main__":
    main()
