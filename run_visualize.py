"""전체 알고리즘 학습 + 개인 보행 분석 시각화 파이프라인.

모델 학습 → 성능 평가 → 모달리티 분석 → 개인 보행 분석 시각화를
한 번에 실행하여 종합 보고서를 생성합니다.
"""

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
from src.validation.report import generate_report
from src.validation.visualize import (
    plot_confusion_matrix,
    plot_confidence_distribution,
    plot_modality_ablation,
    plot_per_class_metrics,
    plot_summary_dashboard,
    plot_training_curves,
)
from src.analysis.visualize_analysis import (
    plot_pressure_heatmap,
    plot_cop_trajectory,
    plot_zone_temporal,
    plot_injury_risk_dashboard,
    plot_gait_profile_deviation,
    plot_trend_dashboard,
    plot_full_analysis_report,
)
from src.analysis import (
    FootZoneAnalyzer,
    PersonalGaitProfiler,
    InjuryRiskEngine,
    CorrektiveFeedbackGenerator,
    LongitudinalTrendTracker,
)


def run_ablation(config: dict, device: torch.device) -> dict:
    """모달리티 제거 실험 (Ablation Study)."""
    checkpoint = torch.load("outputs/best_model.pt", map_location=device, weights_only=False)
    model = MultimodalGaitNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_cfg = config["data"]
    dataset_dict = generate_synthetic_dataset(
        num_samples_per_class=30, num_classes=data_cfg["num_classes"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"], seed=99,
    )
    dataset = MultimodalGaitDataset(
        dataset_dict, sequence_length=data_cfg["sequence_length"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )
    loader = DataLoader(dataset, batch_size=32)

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


def run_evaluation(config: dict, device: torch.device):
    """전체 평가: 예측 + 확률."""
    checkpoint = torch.load("outputs/best_model.pt", map_location=device, weights_only=False)
    model = MultimodalGaitNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_cfg = config["data"]
    dataset_dict = generate_synthetic_dataset(
        num_samples_per_class=30, num_classes=data_cfg["num_classes"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"], seed=99,
    )
    dataset = MultimodalGaitDataset(
        dataset_dict, sequence_length=data_cfg["sequence_length"],
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


def generate_gait_pattern(rng, pattern="normal", T=128, H=16, W=8):
    """다양한 보행 패턴의 합성 압력/IMU 데이터 생성."""
    pressure = rng.rand(T, 1, H, W).astype(np.float32) * 0.3
    imu = rng.randn(6, T).astype(np.float32)
    t = np.linspace(0, 4 * np.pi, T)
    imu[1] += np.sin(t) * 2

    if pattern == "normal":
        pressure[:, 0, 11:16, :] += 0.4
        pressure[:, 0, 3:7, :] += 0.35
        pressure[:, 0, 0:3, :] += 0.2
        pressure[:, 0, 7:11, :] += 0.1
        imu[1] += np.sin(2 * t) * 0.5
    elif pattern == "flat_foot":
        pressure[:, 0, 7:11, :] += 0.5
        pressure[:, 0, 11:16, :] += 0.3
        pressure[:, 0, 3:7, :] += 0.3
        pressure[:, 0, 7:11, 0:4] += 0.3
        imu[0] += np.sin(t + 0.3) * 0.8
    elif pattern == "heel_striker":
        pressure[:, 0, 11:16, :] += 0.8
        pressure[:, 0, 3:7, :] += 0.2
        pressure[:, 0, 0:3, :] += 0.1
    elif pattern == "forefoot_overload":
        pressure[:, 0, 0:7, :] += 0.7
        pressure[:, 0, 11:16, :] += 0.2
    elif pattern == "lateral_shift":
        pressure[:, 0, :, 4:8] += 0.5
        pressure[:, 0, 11:16, :] += 0.3
        imu[0] += np.sin(t) * 1.5

    return pressure, imu


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    output_dir = Path("outputs")
    figures_dir = output_dir / "figures"
    report_dir = output_dir / "report"
    analysis_dir = output_dir / "analysis"
    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================================================================
    # PART 1: 딥러닝 모델 학습 및 평가 시각화
    # ================================================================
    print("=" * 70)
    print("  PART 1: 멀티모달 보행 AI 모델 학습")
    print("=" * 70)
    train(config, output_dir)

    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=False)
    history = checkpoint["history"]

    print("\n학습 곡선 시각화 생성...")
    plot_training_curves(history, figures_dir / "training_curves.png")

    print("\n모델 평가 실행...")
    y_true, y_pred, probs, class_names = run_evaluation(config, device)

    plot_confusion_matrix(y_true, y_pred, class_names, figures_dir / "confusion_matrix.png")
    plot_per_class_metrics(y_true, y_pred, class_names, figures_dir / "per_class_metrics.png")
    plot_confidence_distribution(y_true, y_pred, probs, class_names, figures_dir / "confidence_dist.png")

    print("\n모달리티 Ablation Study:")
    ablation_results = run_ablation(config, device)
    plot_modality_ablation(ablation_results, figures_dir / "modality_ablation.png")

    plot_summary_dashboard(
        history, y_true, y_pred, probs, class_names, ablation_results,
        figures_dir / "dashboard.png",
    )

    print("\n한글 보고서 생성 (3페이지)...")
    model = MultimodalGaitNet(config)
    generate_report(
        history=history, y_true=y_true, y_pred=y_pred, probs=probs,
        class_names=class_names, ablation_results=ablation_results,
        model_params=model.get_num_trainable_params(), save_dir=report_dir,
    )

    # ================================================================
    # PART 2: 개인 맞춤형 보행 분석 시각화
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 2: 개인 맞춤형 보행 분석 시각화")
    print("=" * 70)

    rng = np.random.RandomState(42)
    profiler = PersonalGaitProfiler(16, 8)
    tracker = LongitudinalTrendTracker()

    # 6개 세션 시뮬레이션 (기준선 구축 → 이상 패턴 감지)
    sessions = [
        ("normal", "정상 보행 세션 1"),
        ("normal", "정상 보행 세션 2"),
        ("normal", "정상 보행 세션 3"),
        ("flat_foot", "평발 패턴 감지"),
        ("heel_striker", "뒤꿈치 과부하"),
        ("forefoot_overload", "앞발 과부하"),
    ]

    for i, (pattern, label) in enumerate(sessions):
        print(f"\n  세션 {i + 1}/{len(sessions)}: {label}")
        pressure, imu = generate_gait_pattern(rng, pattern)

        result = plot_full_analysis_report(
            pressure_seq=pressure,
            imu_seq=imu,
            save_dir=analysis_dir / f"session_{i + 1:02d}",
            session_label=label,
            profiler=profiler,
            tracker=tracker,
        )

        # 마지막 세션의 피드백 출력
        if i == len(sessions) - 1:
            print("\n" + result["feedback"].report_kr)

    # ================================================================
    # 최종 트렌드 시각화 (전체 세션 종합)
    # ================================================================
    print("\n" + "=" * 70)
    print("  종합 트렌드 시각화")
    print("=" * 70)
    trend = tracker.analyze_trends(min_sessions=3)
    plot_trend_dashboard(trend, tracker, analysis_dir / "overall_trend.png")
    print(trend.report_kr)

    # ================================================================
    # 결과 요약
    # ================================================================
    metrics = compute_metrics(y_true, y_pred)
    print("\n" + "=" * 70)
    print("  전체 생성 파일 목록")
    print("=" * 70)

    print("\n  [모델 학습 결과]")
    for f in sorted(figures_dir.glob("*.png")):
        print(f"    {f}")

    print("\n  [한글 보고서]")
    for f in sorted(report_dir.glob("*.png")):
        print(f"    {f}")

    print("\n  [개인 보행 분석]")
    for f in sorted(analysis_dir.rglob("*.png")):
        print(f"    {f}")

    total_files = (
        len(list(figures_dir.glob("*.png")))
        + len(list(report_dir.glob("*.png")))
        + len(list(analysis_dir.rglob("*.png")))
    )
    print(f"\n  총 {total_files}개 시각화 파일 생성 완료!")
    print(f"  모델 정확도: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
