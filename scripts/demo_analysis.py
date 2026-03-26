"""개인 맞춤형 보행 분석 시스템 데모."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import (
    FootZoneAnalyzer,
    PersonalGaitProfiler,
    InjuryRiskEngine,
    CorrektiveFeedbackGenerator,
    LongitudinalTrendTracker,
)


def generate_demo_data(rng, pattern="normal"):
    """Generate demo pressure + IMU data with specific gait patterns."""
    T, H, W = 128, 16, 8

    pressure = rng.rand(T, 1, H, W).astype(np.float32) * 0.3

    if pattern == "normal":
        # Balanced pressure distribution
        pressure[:, 0, 11:16, :] += 0.4   # heel
        pressure[:, 0, 3:7, :] += 0.35    # forefoot
        pressure[:, 0, 0:3, :] += 0.2     # toes
        pressure[:, 0, 7:11, :] += 0.1    # midfoot

    elif pattern == "flat_foot":
        # Excessive midfoot contact
        pressure[:, 0, 7:11, :] += 0.5    # high midfoot
        pressure[:, 0, 11:16, :] += 0.3
        pressure[:, 0, 3:7, :] += 0.3
        pressure[:, 0, 7:11, 0:4] += 0.3  # medial shift

    elif pattern == "heel_striker":
        # Excessive heel impact
        pressure[:, 0, 11:16, :] += 0.8   # very high heel
        pressure[:, 0, 3:7, :] += 0.2
        pressure[:, 0, 0:3, :] += 0.1

    elif pattern == "forefoot_overload":
        # Excessive forefoot pressure
        pressure[:, 0, 0:7, :] += 0.7     # high forefoot
        pressure[:, 0, 11:16, :] += 0.2

    # Generate matching IMU data
    imu = rng.randn(6, T).astype(np.float32)
    # Add periodic pattern (simulating walking)
    t = np.linspace(0, 4 * np.pi, T)
    imu[1] += np.sin(t) * 2  # vertical acceleration
    if pattern == "normal":
        imu[1] += np.sin(2 * t) * 0.5  # regular stride
    elif pattern == "flat_foot":
        imu[0] += np.sin(t + 0.3) * 0.8  # lateral sway

    return pressure, imu


def main():
    rng = np.random.RandomState(42)
    profiler = PersonalGaitProfiler(16, 8)
    injury_engine = InjuryRiskEngine(16, 8)
    feedback_gen = CorrektiveFeedbackGenerator()
    tracker = LongitudinalTrendTracker()

    patterns = [
        ("normal", "정상 보행"),
        ("normal", "정상 보행 (세션 2)"),
        ("normal", "정상 보행 (세션 3)"),
        ("flat_foot", "평발 패턴"),
        ("heel_striker", "뒤꿈치 과부하"),
        ("forefoot_overload", "앞발 과부하"),
    ]

    print("=" * 70)
    print("  🦶 개인 맞춤형 보행 분석 시스템 데모")
    print("=" * 70)

    for i, (pattern, name) in enumerate(patterns):
        pressure, imu = generate_demo_data(rng, pattern)

        print(f"\n{'#' * 70}")
        print(f"  세션 {i + 1}: {name}")
        print(f"{'#' * 70}")

        # 1. Extract features
        features = profiler.extract_session_features(pressure, imu)
        print(f"\n  [특성 추출] 추출된 특성 수: {len(features)}")
        print(f"    내외측 지수: {features['ml_index']:.3f}")
        print(f"    전후방 지수: {features['ap_index']:.3f}")
        print(f"    아치 지수: {features['arch_index']:.3f}")
        print(f"    COP 흔들림: {features['cop_sway']:.4f}")
        if "cadence" in features:
            print(f"    보행률: {features['cadence']:.1f} steps/min")
            print(f"    보폭 규칙성: {features['stride_regularity']:.3f}")
            print(f"    좌우 대칭성: {features['step_symmetry']:.3f}")

        # 2. Update baseline and check deviations
        profiler.update_baseline(features)
        deviation = profiler.compute_deviations(features)
        if deviation.alerts:
            print(f"\n  [기준 대비 변화] {len(deviation.alerts)}개 알림:")
            for alert in deviation.alerts:
                print(f"    [{alert['severity']}] {alert['message']}")
        elif profiler.baseline.num_sessions >= 2:
            print(f"\n  [기준 대비 변화] 평소 패턴과 유사합니다.")

        # 3. Injury risk assessment
        injury_report = injury_engine.assess_risk(pressure)
        print(f"\n  [부상 위험 평가] 전체 위험도: {injury_report.overall_risk:.2f}")
        for risk in sorted(injury_report.risks, key=lambda r: -r.risk_score):
            bar = "█" * int(risk.risk_score * 10) + "░" * (10 - int(risk.risk_score * 10))
            print(f"    {risk.korean_name:12s} [{bar}] {risk.risk_score:.2f} ({risk.severity})")

        # 4. Track for trends
        tracker.add_session(features, injury_report.overall_risk, deviation.overall_deviation)

        # 5. Generate personalized feedback (for last session)
        if i == len(patterns) - 1:
            print("\n" + "=" * 70)
            print("  최종 세션 맞춤형 피드백")
            print("=" * 70)
            feedback = feedback_gen.generate(injury_report, deviation, profiler.baseline)
            print(feedback.report_kr)

    # 6. Trend analysis
    print("\n")
    trend = tracker.analyze_trends(min_sessions=3)
    print(trend.report_kr)


if __name__ == "__main__":
    main()
