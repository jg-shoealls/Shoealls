"""Plot training curves from a saved checkpoint."""

import sys
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(checkpoint_path: str, out_path: str = "outputs/training_curves.png"):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = ckpt["history"]

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    best_epoch = ckpt["epoch"]
    best_val = ckpt["val_accuracy"]
    fig.suptitle(
        f"Best epoch: {best_epoch}  |  Best val acc: {best_val:.4f}  |  "
        f"num_classes: {ckpt['config']['data']['num_classes']}",
        fontsize=11,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/run_001/best_model.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/training_curves.png"
    plot(path, out)
