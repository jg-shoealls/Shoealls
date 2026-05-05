"""Full pipeline: train model, run validation, generate all visualizations."""

from pathlib import Path

import torch
import yaml

from src.models.multimodal_gait_net import MultimodalGaitNet
from src.training.evaluation import run_ablation, run_evaluation
from src.training.train import train
from src.utils.metrics import compute_metrics
from src.validation.report import generate_report
from src.validation.visualize import (
    plot_confusion_matrix,
    plot_confidence_distribution,
    plot_modality_ablation,
    plot_per_class_metrics,
    plot_summary_dashboard,
    plot_training_curves,
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

    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=True)
    history = checkpoint["history"]

    plot_training_curves(history, figures_dir / "training_curves.png")

    # Step 3: Full evaluation
    y_true, y_pred, probs, class_names = run_evaluation(config, device)

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

    # Step 6: Generate meeting report (Korean)
    print("\n" + "=" * 60)
    print("STEP 6: Generating Meeting Report (Korean)")
    print("=" * 60)

    report_dir = output_dir / "report"
    model = MultimodalGaitNet(config)
    generate_report(
        history=history,
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        class_names=class_names,
        ablation_results=ablation_results,
        model_params=model.get_num_trainable_params(),
        save_dir=report_dir,
    )

    # Print summary
    metrics = compute_metrics(y_true, y_pred)
    print("\n" + "=" * 60)
    print("ALL DONE! Generated files:")
    print("=" * 60)
    for f in sorted(figures_dir.glob("*.png")):
        print(f"  {f}")
    print()
    for f in sorted(report_dir.glob("*.png")):
        print(f"  {f}")
    print(f"\nFinal Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
