"""Visualization module for multimodal gait analysis results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ── Color palette ──────────────────────────────────────────────────────
COLORS = {
    "normal": "#4CAF50",
    "antalgic": "#FF9800",
    "ataxic": "#F44336",
    "parkinsonian": "#9C27B0",
}
MODALITY_COLORS = {"IMU": "#2196F3", "Pressure": "#FF5722", "Skeleton": "#009688"}


def plot_training_curves(history: dict, save_path: Path):
    """Plot training & validation loss/accuracy curves.

    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: Where to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "o-", label="Train", color="#2196F3", markersize=3)
    ax.plot(epochs, history["val_loss"], "s-", label="Val", color="#F44336", markersize=3)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Accuracy curve
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], "o-", label="Train", color="#2196F3", markersize=3)
    ax.plot(epochs, history["val_acc"], "s-", label="Val", color="#F44336", markersize=3)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
    normalize: bool = True,
):
    """Plot confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Where to save the figure.
        normalize: Whether to normalize by true label count.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    else:
        cm_norm = cm.astype(float)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Ratio" if normalize else "Count", fontsize=11)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")

    # Annotate cells with both count and percentage
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            pct = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if pct > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.0%})", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_confidence_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    save_path: Path,
):
    """Plot prediction confidence distribution by correctness and per class.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        probs: Softmax probabilities of shape (N, num_classes).
        class_names: List of class names.
        save_path: Where to save the figure.
    """
    max_probs = probs.max(axis=1)
    correct = y_true == y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: correct vs incorrect confidence
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    if correct.any():
        ax.hist(max_probs[correct], bins=bins, alpha=0.7, label="Correct",
                color="#4CAF50", edgecolor="white")
    if (~correct).any():
        ax.hist(max_probs[~correct], bins=bins, alpha=0.7, label="Incorrect",
                color="#F44336", edgecolor="white")
    ax.set_xlabel("Prediction Confidence", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Confidence: Correct vs Incorrect", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: per-class confidence box plot
    ax = axes[1]
    class_confs = []
    for i in range(len(class_names)):
        mask = y_true == i
        class_confs.append(max_probs[mask])

    bp = ax.boxplot(class_confs, labels=class_names, patch_artist=True)
    colors = [COLORS.get(n, "#999") for n in class_names]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Prediction Confidence", fontsize=12)
    ax.set_title("Per-Class Confidence", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
):
    """Plot per-class precision, recall, F1 as grouped bar chart."""
    from sklearn.metrics import precision_recall_fscore_support

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(class_names))
    width = 0.22

    bars1 = ax.bar(x - width, prec, width, label="Precision", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, rec, width, label="Recall", color="#FF9800", alpha=0.85)
    bars3 = ax.bar(x + width, f1, width, label="F1 Score", color="#4CAF50", alpha=0.85)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Support counts
    for i, s in enumerate(support):
        ax.text(i, -0.08, f"n={s}", ha="center", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Performance Metrics", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_modality_ablation(ablation_results: dict, save_path: Path):
    """Plot modality ablation study results.

    Args:
        ablation_results: Dict mapping modality combination name to accuracy.
            e.g. {"IMU only": 0.72, "Pressure only": 0.65, ..., "All (Fusion)": 0.98}
        save_path: Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(ablation_results.keys())
    accs = list(ablation_results.values())
    colors = []
    for name in names:
        if "All" in name or "Fusion" in name:
            colors.append("#4CAF50")
        elif "+" in name:
            colors.append("#FF9800")
        else:
            colors.append("#2196F3")

    bars = ax.barh(range(len(names)), accs, color=colors, alpha=0.85, edgecolor="white")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}", va="center", fontsize=11, fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Modality Ablation Study", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_summary_dashboard(
    history: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    ablation_results: dict | None,
    save_path: Path,
):
    """Generate a single-page summary dashboard with all key visualizations.

    This is the main entry point for result visualization.
    """
    has_ablation = ablation_results is not None
    nrows = 3 if has_ablation else 2
    fig = plt.figure(figsize=(20, 6 * nrows))
    gs = gridspec.GridSpec(nrows, 3, hspace=0.35, wspace=0.3)

    # ── Row 1: Training curves + Confusion matrix ──────────────────────
    # Training loss
    ax = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "o-", label="Train", color="#2196F3", markersize=2)
    ax.plot(epochs, history["val_loss"], "s-", label="Val", color="#F44336", markersize=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Training accuracy
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, history["train_acc"], "o-", label="Train", color="#2196F3", markersize=2)
    ax.plot(epochs, history["val_acc"], "s-", label="Val", color="#F44336", markersize=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Confusion matrix
    ax = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels([n[:6] for n in class_names], rotation=45, ha="right")
    ax.set_yticklabels([n[:6] for n in class_names])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontweight="bold")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                    ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    # ── Row 2: Per-class metrics + Confidence ──────────────────────────
    from sklearn.metrics import precision_recall_fscore_support

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0,
    )

    # Per-class bars
    ax = fig.add_subplot(gs[1, 0:2])
    x = np.arange(len(class_names))
    w = 0.22
    ax.bar(x - w, prec, w, label="Precision", color="#2196F3", alpha=0.85)
    ax.bar(x, rec, w, label="Recall", color="#FF9800", alpha=0.85)
    ax.bar(x + w, f1, w, label="F1", color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.15)

    # Confidence distribution
    ax = fig.add_subplot(gs[1, 2])
    max_probs = probs.max(axis=1)
    correct = y_true == y_pred
    bins = np.linspace(0, 1, 21)
    if correct.any():
        ax.hist(max_probs[correct], bins=bins, alpha=0.7, label="Correct", color="#4CAF50")
    if (~correct).any():
        ax.hist(max_probs[~correct], bins=bins, alpha=0.7, label="Incorrect", color="#F44336")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Row 3: Ablation (optional) ─────────────────────────────────────
    if has_ablation:
        ax = fig.add_subplot(gs[2, :])
        names = list(ablation_results.keys())
        accs = list(ablation_results.values())
        bar_colors = []
        for name in names:
            if "All" in name or "Fusion" in name:
                bar_colors.append("#4CAF50")
            elif "+" in name:
                bar_colors.append("#FF9800")
            else:
                bar_colors.append("#2196F3")
        bars = ax.barh(range(len(names)), accs, color=bar_colors, alpha=0.85)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.1%}", va="center", fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Accuracy")
        ax.set_title("Modality Ablation Study", fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

    fig.suptitle("Multimodal Gait Analysis - Results Dashboard",
                 fontsize=18, fontweight="bold", y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dashboard: {save_path}")
