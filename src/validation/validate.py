"""Validation and initial verification of the multimodal gait model."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalGaitDataset
from src.data.synthetic import generate_synthetic_dataset
from src.models.multimodal_gait_net import MultimodalGaitNet
from src.utils.metrics import compute_metrics


def run_validation(config: dict, checkpoint_path: str):
    """Run full validation on a trained model checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MultimodalGaitNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Checkpoint val accuracy: {checkpoint['val_accuracy']:.4f}")

    # Generate test data
    data_cfg = config["data"]
    dataset_dict = generate_synthetic_dataset(
        num_samples_per_class=30,
        num_classes=data_cfg["num_classes"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
        seed=99,  # Different seed for validation
    )

    dataset = MultimodalGaitDataset(
        dataset_dict,
        sequence_length=data_cfg["sequence_length"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )

    loader = DataLoader(dataset, batch_size=32)

    # Evaluate
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)

            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    metrics = compute_metrics(all_labels, all_preds)

    class_names = dataset_dict["class_names"]
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1 Score:    {metrics['f1_macro']:.4f}")
    print(f"Macro Precision:   {metrics['precision']:.4f}")
    print(f"Macro Recall:      {metrics['recall']:.4f}")

    print(f"\nPer-class Accuracy:")
    cm = metrics["confusion_matrix"]
    for i, name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {name:15s}: {class_acc:.4f} ({cm[i, i]}/{cm[i].sum()})")

    print(f"\nConfusion Matrix:")
    header = "            " + " ".join(f"{n[:6]:>7s}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = " ".join(f"{cm[i, j]:7d}" for j in range(len(class_names)))
        print(f"  {name[:10]:10s} {row}")

    # Confidence analysis
    correct_mask = all_preds == all_labels
    print(f"\nConfidence Analysis:")
    print(f"  Mean confidence (correct):   {all_probs.max(axis=1)[correct_mask].mean():.4f}")
    if (~correct_mask).any():
        print(f"  Mean confidence (incorrect): {all_probs.max(axis=1)[~correct_mask].mean():.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Validate multimodal gait model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_validation(config, args.checkpoint)


if __name__ == "__main__":
    main()
