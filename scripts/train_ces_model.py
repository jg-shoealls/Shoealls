"""Train a CES gait-state LSTM from the Shoealls event TSV.

Example:
    python scripts/train_ces_model.py --data CES.csv --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.data.ces_processor import CESDataProcessor
from src.models.gait_lstm import GaitLSTM, GaitWindowDataset, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Shoealls CES LSTM model")
    parser.add_argument("--data", type=Path, default=Path("CES.csv"), help="Path to tab-separated CES.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("outputs/ces/gait_lstm_ces.pt"))
    return parser.parse_args()


def evaluate(model: GaitLSTM, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            label = batch["label"].to(device)
            logits, _, _ = model(x)
            correct += int((logits.argmax(dim=-1) == label).sum().item())
            total += int(label.numel())
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    processor = CESDataProcessor(args.data)
    processed = processor.preprocess()

    dataset = GaitWindowDataset(
        features=processed.features,
        labels=processed.labels,
        fall_flags=processed.fall_flags,
        window_size=args.window_size,
    )

    train_size = max(1, int(len(dataset) * 0.8))
    val_size = len(dataset) - train_size
    if val_size == 0:
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitLSTM(
        input_dim=processed.features.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(processed.label_encoder.classes_),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        "Training CES LSTM "
        f"rows={len(processed.dataframe)} windows={len(dataset)} "
        f"classes={list(processed.label_encoder.classes_)} device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_accuracy = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss={train_result.loss:.4f} "
            f"train_acc={train_result.accuracy:.3f} "
            f"val_acc={val_accuracy:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": processed.features.shape[1],
            "classes": list(processed.label_encoder.classes_),
            "event_types": list(processed.event_encoder.classes_),
            "window_size": args.window_size,
        },
        args.output,
    )
    print(f"Saved model checkpoint: {args.output}")


if __name__ == "__main__":
    main()
