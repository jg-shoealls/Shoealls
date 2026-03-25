"""Full pipeline: train model, run validation, generate all visualizations."""

from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalGaitDataset
from src.data.synthetic import generate_synthetic_dataset
from src.models.multimodal_gait_net import MultimodalGaitNet
from src.training.train import train
from src.utils.metrics import compute_metrics
from src.validation.visualize import (
    plot_confusion_matrix,
    plot_confidence_distribution,
    plot_modality_ablation,
    plot_per_class_metrics,
    plot_summary_dashboard,
    plot_training_curves,
)


def run_ablation(config: dict, device: torch.device) -> dict:
    """Run modality ablation study with zero-masking."""
    checkpoint = torch.load("outputs/best_model.pt", map_location=device, weights_only=False)
    model = MultimodalGaitNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_cfg = config["data"]
    dataset_dict = generate_synthetic_dataset(
        num_samples_per_class=30,
        num_classes=data_cfg["num_classes"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
        seed=99,
    )
    dataset = MultimodalGaitDataset(
        dataset_dict,
        sequence_length=data_cfg["sequence_length"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )
    loader = DataLoader(dataset, batch_size=32)

    # Define masking strategies: which modalities to zero out
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
    for name, mask in strategies.items():
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("label")

                # Zero-mask excluded modalities
                for key in mask:
                    batch[key] = torch.zeros_like(batch[key])

                logits = model(batch)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = (all_preds == all_labels).mean()
        results[name] = acc
        print(f"  {name:25s}: {acc:.4f}")

    return results


def run_full_evaluation(config: dict, device: torch.device):
    """Run full evaluation and collect predictions + probabilities."""
    checkpoint = torch.load("outputs/best_model.pt", map_location=device, weights_only=False)
    model = MultimodalGaitNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_cfg = config["data"]
    dataset_dict = generate_synthetic_dataset(
        num_samples_per_class=30,
        num_classes=data_cfg["num_classes"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
        seed=99,
    )
    dataset = MultimodalGaitDataset(
        dataset_dict,
        sequence_length=data_cfg["sequence_length"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )
    loader = DataLoader(dataset, batch_size=32)

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

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        dataset_dict["class_names"],
    )


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    output_dir = Path("outputs")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Train
    print("=" * 60)
    print("STEP 1: Training")
    print("=" * 60)
    train(config, output_dir)

    # Step 2: Load history and generate training curve plots
    print("\n" + "=" * 60)
    print("STEP 2: Generating Visualizations")
    print("=" * 60)

    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=False)
    history = checkpoint["history"]

    plot_training_curves(history, figures_dir / "training_curves.png")

    # Step 3: Full evaluation
    y_true, y_pred, probs, class_names = run_full_evaluation(config, device)

    plot_confusion_matrix(y_true, y_pred, class_names, figures_dir / "confusion_matrix.png")
    plot_per_class_metrics(y_true, y_pred, class_names, figures_dir / "per_class_metrics.png")
    plot_confidence_distribution(
        y_true, y_pred, probs, class_names, figures_dir / "confidence_dist.png"
    )

    # Step 4: Ablation study
    print("\nModality Ablation Study:")
    ablation_results = run_ablation(config, device)
    plot_modality_ablation(ablation_results, figures_dir / "modality_ablation.png")

    # Step 5: Summary dashboard
    plot_summary_dashboard(
        history, y_true, y_pred, probs, class_names, ablation_results,
        figures_dir / "dashboard.png",
    )

    # Print summary
    metrics = compute_metrics(y_true, y_pred)
    print("\n" + "=" * 60)
    print("ALL DONE! Generated figures:")
    print("=" * 60)
    for f in sorted(figures_dir.glob("*.png")):
        print(f"  {f}")
    print(f"\nFinal Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
