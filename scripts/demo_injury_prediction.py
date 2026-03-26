"""비정상 보행 패턴 감지 및 부상 위험 예측 데모.

12개 이상 보행 패턴 감지 + ML 기반 9개 부상 유형 예측.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.injury_predictor import InjuryRiskPredictor


def main():
    print("=" * 70)
    print("  슈올즈 AI — 비정상 보행 패턴 감지 & 부상 위험 예측 데모")
    print("=" * 70)

    # === 1. ML 예측기 학습 ===
    print("\n[1단계] ML 부상 예측기 학습 (9개 부상 시나리오)")
    print("-" * 50)
    predictor = InjuryRiskPredictor(n_estimators=100)
    metrics = predictor.train()
    print(f"  학습 정확도: {metrics.accuracy:.1%}")
    print(f"  교차검증: {metrics.cv_accuracy_mean:.1%} (±{metrics.cv_accuracy_std:.1%})")
    print(f"  F1 Score: {metrics.f1_macro:.4f}")

    # 특성 중요도 Top 5
    sorted_feats = sorted(metrics.feature_importance.items(), key=lambda x: -x[1])
    print("\n  [특성 중요도 Top 5]")
    from src.analysis.injury_predictor import FEATURE_KOREAN
    for rank, (feat, imp) in enumerate(sorted_feats[:5], 1):
        kr = FEATURE_KOREAN.get(feat, feat)
        print(f"  {rank}. {kr:14s} {imp:.3f}")

    # === 2. 다양한 환자 시뮬레이션 ===
    patients = {
        "정상 보행 (25세 여성, 건강)": {
            "gait_speed": 1.25, "cadence": 118, "stride_regularity": 0.88,
            "step_symmetry": 0.93, "cop_sway": 0.035, "ml_index": 0.03,
            "arch_index": 0.24, "acceleration_rms": 1.6,
            "zone_heel_medial_mean": 0.32, "zone_heel_lateral_mean": 0.30,
            "zone_forefoot_medial_mean": 0.33, "zone_forefoot_lateral_mean": 0.30,
            "zone_toes_mean": 0.18, "zone_midfoot_medial_mean": 0.10,
            "zone_midfoot_lateral_mean": 0.08,
        },
        "전족부 과부하 러너 (32세 남성)": {
            "gait_speed": 1.15, "cadence": 125, "stride_regularity": 0.82,
            "step_symmetry": 0.90, "cop_sway": 0.05, "ml_index": 0.04,
            "arch_index": 0.20, "acceleration_rms": 2.2,
            "zone_heel_medial_mean": 0.08, "zone_heel_lateral_mean": 0.06,
            "zone_forefoot_medial_mean": 0.48, "zone_forefoot_lateral_mean": 0.42,
            "zone_toes_mean": 0.25, "zone_midfoot_medial_mean": 0.05,
            "zone_midfoot_lateral_mean": 0.04,
        },
        "발목 불안정 (28세 여성, 잦은 염좌)": {
            "gait_speed": 1.05, "cadence": 112, "stride_regularity": 0.72,
            "step_symmetry": 0.82, "cop_sway": 0.08, "ml_index": 0.14,
            "arch_index": 0.30, "acceleration_rms": 1.4,
            "zone_heel_medial_mean": 0.20, "zone_heel_lateral_mean": 0.35,
            "zone_forefoot_medial_mean": 0.30, "zone_forefoot_lateral_mean": 0.40,
            "zone_toes_mean": 0.12, "zone_midfoot_medial_mean": 0.08,
            "zone_midfoot_lateral_mean": 0.14,
        },
        "무릎 과부하 (45세 남성, 과체중)": {
            "gait_speed": 0.95, "cadence": 108, "stride_regularity": 0.74,
            "step_symmetry": 0.84, "cop_sway": 0.06, "ml_index": 0.08,
            "arch_index": 0.22, "acceleration_rms": 2.4,
            "zone_heel_medial_mean": 0.38, "zone_heel_lateral_mean": 0.35,
            "zone_forefoot_medial_mean": 0.35, "zone_forefoot_lateral_mean": 0.32,
            "zone_toes_mean": 0.15, "zone_midfoot_medial_mean": 0.12,
            "zone_midfoot_lateral_mean": 0.10,
        },
        "낙상 고위험 (75세 여성)": {
            "gait_speed": 0.55, "cadence": 78, "stride_regularity": 0.40,
            "step_symmetry": 0.68, "cop_sway": 0.13, "ml_index": 0.18,
            "arch_index": 0.32, "acceleration_rms": 0.65,
            "zone_heel_medial_mean": 0.22, "zone_heel_lateral_mean": 0.28,
            "zone_forefoot_medial_mean": 0.35, "zone_forefoot_lateral_mean": 0.32,
            "zone_toes_mean": 0.10, "zone_midfoot_medial_mean": 0.12,
            "zone_midfoot_lateral_mean": 0.14,
        },
        "절뚝거림 보행 (38세 남성, 좌측 발목 부상)": {
            "gait_speed": 0.65, "cadence": 88, "stride_regularity": 0.52,
            "step_symmetry": 0.52, "cop_sway": 0.09, "ml_index": 0.20,
            "arch_index": 0.26, "acceleration_rms": 0.80,
            "zone_heel_medial_mean": 0.15, "zone_heel_lateral_mean": 0.32,
            "zone_forefoot_medial_mean": 0.45, "zone_forefoot_lateral_mean": 0.30,
            "zone_toes_mean": 0.08, "zone_midfoot_medial_mean": 0.07,
            "zone_midfoot_lateral_mean": 0.14,
        },
    }

    for patient_name, features in patients.items():
        print(f"\n{'#' * 70}")
        print(f"  환자: {patient_name}")
        print(f"{'#' * 70}")

        report = predictor.predict_comprehensive(features)
        print(report.summary_kr)

    print(f"\n{'=' * 70}")
    print("  ※ 슈올즈 AI: 보행 패턴 분석으로 부상을 예방합니다.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
