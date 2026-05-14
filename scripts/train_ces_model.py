"""Train and validate a CES gait-state LSTM from Shoealls event TSV data.

The default validation path is intentionally conservative:
  - temporal split instead of random adjacent-window split
  - sensor-only features by default to reduce label leakage from event_type
  - per-class precision/recall/F1 and confusion matrix reporting

Example:
    $env:PYTHONPATH=(Get-Location).Path
    python scripts/train_ces_model.py --data CES.csv --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.data.ces_processor import CESDataProcessor
from src.models.gait_lstm import GaitLSTM, GaitWindowDataset, train_one_epoch

FEATURE_SETS = {
    "sensor_only": ["speed", "tilt", "fe_event_value"],
    "with_event_type": ["speed", "tilt", "fe_event_value", "event_type_encoded"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Shoealls CES LSTM model")
    parser.add_argument("--data", type=Path, default=Path("CES.csv"), help="Path to tab-separated CES.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split", choices=["temporal", "random"], default="temporal")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="sensor_only")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/ces/gait_lstm_ces.pt"))
    return parser.parse_args()


def build_features(processor: CESDataProcessor, feature_set: str) -> tuple[np.ndarray, list[str]]:
    if processor.df is None:
        raise ValueError("processor.preprocess() must be called before build_features")

    feature_names = FEATURE_SETS[feature_set]
    missing = [name for name in feature_names if name not in processor.df.columns]
    if missing:
        raise ValueError(f"preprocessed dataframe is missing feature columns: {missing}")

    return processor.df[feature_names].to_numpy(dtype=np.float32), feature_names


def make_temporal_datasets(
    features: np.ndarray,
    labels: np.ndarray,
    fall_flags: np.ndarray,
    window_size: int,
    stride: int,
    val_ratio: float,
) -> tuple[GaitWindowDataset, GaitWindowDataset]:
    split_row = int(len(features) * (1.0 - val_ratio))
    split_row = max(window_size, min(split_row, len(features) - window_size))

    train_dataset = GaitWindowDataset(
        features=features[:split_row],
        labels=labels[:split_row],
        fall_flags=fall_flags[:split_row],
        window_size=window_size,
        stride=stride,
    )
    val_dataset = GaitWindowDataset(
        features=features[split_row:],
        labels=labels[split_row:],
        fall_flags=fall_flags[split_row:],
        window_size=window_size,
        stride=stride,
    )
    return train_dataset, val_dataset


def make_random_datasets(
    features: np.ndarray,
    labels: np.ndarray,
    fall_flags: np.ndarray,
    window_size: int,
    stride: int,
    val_ratio: float,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    dataset = GaitWindowDataset(features, labels, fall_flags, window_size=window_size, stride=stride)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("not enough windows for random train/validation split")
    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = np.sqrt(counts.sum() / (num_classes * counts))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            matrix[true, pred] += 1
    return matrix


def summarize_metrics(matrix: np.ndarray, class_names: list[str]) -> dict[str, object]:
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    weighted_f1_sum = 0.0
    total_support = int(matrix.sum())
    correct = int(np.trace(matrix))

    for idx, name in enumerate(class_names):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - matrix[idx, idx])
        fn = float(matrix[idx, :].sum() - matrix[idx, idx])
        support = float(matrix[idx, :].sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        weighted_f1_sum += f1 * support

    return {
        "accuracy": correct / max(total_support, 1),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "weighted_f1": weighted_f1_sum / max(total_support, 1),
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(
    model: GaitLSTM,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    fall_true = 0
    fall_ai_detected = 0
    fall_rule_detected = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        label = batch["label"].to(device)
        fall_flag = batch["fall_flag"].to(device)
        logits, fall_score, _ = model(x)
        pred = logits.argmax(dim=-1)

        y_true.extend(label.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        predicted_fall = fall_score.squeeze(-1) >= 0.65
        fall_bool = fall_flag.bool()
        fall_true += int(fall_bool.sum().item())
        fall_ai_detected += int((predicted_fall & fall_bool).sum().item())
        fall_rule_detected += int(fall_bool.sum().item())

    matrix = build_confusion_matrix(y_true, y_pred, len(class_names))
    summary = summarize_metrics(matrix, class_names)
    return {
        "accuracy": summary["accuracy"],
        "macro_f1": summary["macro_f1"],
        "weighted_f1": summary["weighted_f1"],
        "fall_ai_recall": fall_ai_detected / max(fall_true, 1),
        "fall_rule_recall": fall_rule_detected / max(fall_true, 1),
        "per_class": summary["per_class"],
        "confusion_matrix": matrix,
    }


def print_label_distribution(name: str, dataset: torch.utils.data.Dataset, class_names: list[str]) -> None:
    counts = np.zeros(len(class_names), dtype=np.int64)
    fall_windows = 0
    for idx in range(len(dataset)):
        item = dataset[idx]
        counts[int(item["label"].item())] += 1
        fall_windows += int(item["fall_flag"].item())

    parts = ", ".join(f"{class_names[i]}={counts[i]}" for i in range(len(class_names)))
    print(f"{name} windows={len(dataset)} labels=[{parts}] fall_windows={fall_windows}")


def print_final_metrics(metrics: dict[str, object], class_names: list[str]) -> None:
    print(
        "Validation summary "
        f"acc={metrics['accuracy']:.3f} "
        f"macro_f1={metrics['macro_f1']:.3f} "
        f"weighted_f1={metrics['weighted_f1']:.3f} "
        f"fall_rule_recall={metrics['fall_rule_recall']:.3f} "
        f"fall_ai_recall={metrics['fall_ai_recall']:.3f}"
    )

    report = metrics["per_class"]
    assert isinstance(report, dict)
    print("Per-class metrics:")
    for name in class_names:
        row = report[name]
        print(
            f"  {name}: "
            f"precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} "
            f"f1={row['f1']:.3f} "
            f"support={int(row['support'])}"
        )

    matrix = metrics["confusion_matrix"]
    assert isinstance(matrix, np.ndarray)
    print("Confusion matrix rows=true cols=pred:")
    print("  " + " ".join(f"{name[:7]:>7}" for name in class_names))
    for name, row in zip(class_names, matrix):
        print(f"  {name[:7]:>7} " + " ".join(f"{int(value):7d}" for value in row))


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.val_ratio <= 0.5:
        raise ValueError("--val-ratio must be between 0.05 and 0.5")

    processor = CESDataProcessor(args.data)
    processed = processor.preprocess()
    features, feature_names = build_features(processor, args.feature_set)
    labels = processed.labels
    fall_flags = processed.fall_flags
    class_names = list(processed.label_encoder.classes_)

    if args.split == "temporal":
        train_dataset, val_dataset = make_temporal_datasets(
            features,
            labels,
            fall_flags,
            args.window_size,
            args.stride,
            args.val_ratio,
        )
    else:
        train_dataset, val_dataset = make_random_datasets(
            features,
            labels,
            fall_flags,
            args.window_size,
            args.stride,
            args.val_ratio,
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitLSTM(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(class_names),
    ).to(device)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        class_weights = None
    else:
        class_weights = compute_class_weights(labels, len(class_names)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        "Training CES LSTM "
        f"rows={len(processed.dataframe)} classes={class_names} "
        f"features={feature_names} split={args.split} device={device}"
    )
    if class_weights is not None:
        print(f"class_weights={[round(float(v), 4) for v in class_weights.cpu()]}")
    print_label_distribution("train", train_dataset, class_names)
    print_label_distribution("val", val_dataset, class_names)

    best_macro_f1 = -1.0
    best_state = None
    best_metrics: dict[str, object] | None = None

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, device, class_names)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss={train_result.loss:.4f} "
            f"train_acc={train_result.accuracy:.3f} "
            f"val_acc={metrics['accuracy']:.3f} "
            f"val_macro_f1={metrics['macro_f1']:.3f} "
            f"fall_rule_recall={metrics['fall_rule_recall']:.3f} "
            f"fall_ai_recall={metrics['fall_ai_recall']:.3f}"
        )

        if float(metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(metrics["macro_f1"])
            best_metrics = metrics
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    print_final_metrics(best_metrics, class_names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": features.shape[1],
            "classes": class_names,
            "event_types": list(processed.event_encoder.classes_),
            "feature_names": feature_names,
            "window_size": args.window_size,
            "stride": args.stride,
            "split": args.split,
            "best_macro_f1": best_macro_f1,
        },
        args.output,
    )
    print(f"Saved best model checkpoint: {args.output}")


if __name__ == "__main__":
    main()
