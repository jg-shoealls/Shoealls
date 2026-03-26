"""보행 기반 질환 예측 및 진단 데모."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.analysis.biomarkers import GaitBiomarkerExtractor
from src.analysis.disease_predictor import DiseaseRiskPredictor
from src.analysis.disease_classifier import GaitDiseaseClassifier, FEATURE_NAMES


def main():
    print("=" * 70)
    print("  보행 데이터 기반 질환 위험 예측 및 조기 진단 데모")
    print("=" * 70)

    # === 1. ML 분류기 학습 ===
    print("\n[1단계] ML 분류기 학습 (7개 질환 패턴)")
    print("-" * 50)
    clf = GaitDiseaseClassifier(n_estimators=100)
    metrics = clf.train()
    print(f"  학습 정확도: {metrics.accuracy:.1%}")
    print(f"  교차검증: {metrics.cv_accuracy_mean:.1%} (±{metrics.cv_accuracy_std:.1%})")
    print(f"  F1 Score: {metrics.f1_macro:.4f}")
    print()
    print(clf.get_feature_importance_report())

    # === 2. 다양한 환자 시뮬레이션 ===
    patients = {
        "정상 성인 (30대 남성)": {
            "gait_speed": 1.25, "cadence": 118, "stride_regularity": 0.88,
            "step_symmetry": 0.93, "cop_sway": 0.035, "ml_index": 0.03,
            "arch_index": 0.24, "acceleration_rms": 1.6,
            "zone_heel_medial_mean": 0.32, "zone_heel_lateral_mean": 0.30,
            "zone_forefoot_medial_mean": 0.33, "zone_forefoot_lateral_mean": 0.30,
            "zone_toes_mean": 0.18, "zone_midfoot_medial_mean": 0.10,
            "zone_midfoot_lateral_mean": 0.08,
        },
        "파킨슨 의심 (65세 남성)": {
            "gait_speed": 0.65, "cadence": 148, "stride_regularity": 0.42,
            "step_symmetry": 0.78, "cop_sway": 0.065, "ml_index": 0.08,
            "arch_index": 0.23, "acceleration_rms": 0.9,
            "zone_heel_medial_mean": 0.28, "zone_heel_lateral_mean": 0.25,
            "zone_forefoot_medial_mean": 0.38, "zone_forefoot_lateral_mean": 0.35,
            "zone_toes_mean": 0.12, "zone_midfoot_medial_mean": 0.10,
            "zone_midfoot_lateral_mean": 0.10,
        },
        "뇌졸중 후유증 (58세 여성)": {
            "gait_speed": 0.50, "cadence": 82, "stride_regularity": 0.52,
            "step_symmetry": 0.55, "cop_sway": 0.085, "ml_index": 0.22,
            "arch_index": 0.27, "acceleration_rms": 1.0,
            "zone_heel_medial_mean": 0.18, "zone_heel_lateral_mean": 0.35,
            "zone_forefoot_medial_mean": 0.42, "zone_forefoot_lateral_mean": 0.28,
            "zone_toes_mean": 0.10, "zone_midfoot_medial_mean": 0.08,
            "zone_midfoot_lateral_mean": 0.12,
        },
        "당뇨 신경병증 의심 (62세 남성)": {
            "gait_speed": 0.88, "cadence": 102, "stride_regularity": 0.58,
            "step_symmetry": 0.84, "cop_sway": 0.09, "ml_index": 0.06,
            "arch_index": 0.40, "acceleration_rms": 1.2,
            "zone_heel_medial_mean": 0.15, "zone_heel_lateral_mean": 0.12,
            "zone_forefoot_medial_mean": 0.45, "zone_forefoot_lateral_mean": 0.42,
            "zone_toes_mean": 0.20, "zone_midfoot_medial_mean": 0.12,
            "zone_midfoot_lateral_mean": 0.10,
        },
        "소뇌 실조 의심 (45세 여성)": {
            "gait_speed": 0.75, "cadence": 95, "stride_regularity": 0.38,
            "step_symmetry": 0.72, "cop_sway": 0.12, "ml_index": 0.15,
            "arch_index": 0.28, "acceleration_rms": 1.3,
            "zone_heel_medial_mean": 0.28, "zone_heel_lateral_mean": 0.30,
            "zone_forefoot_medial_mean": 0.32, "zone_forefoot_lateral_mean": 0.35,
            "zone_toes_mean": 0.12, "zone_midfoot_medial_mean": 0.10,
            "zone_midfoot_lateral_mean": 0.12,
        },
    }

    predictor = DiseaseRiskPredictor()

    for patient_name, features in patients.items():
        print(f"\n{'#' * 70}")
        print(f"  환자: {patient_name}")
        print(f"{'#' * 70}")

        # 규칙 기반 위험도 평가
        report = predictor.predict(features)
        print(report.summary_kr)

        # ML 분류기 예측
        print("\n  [ML 분류기 예측 결과]")
        clf_result = clf.predict(features)
        print(f"  예측 질환: {clf_result.predicted_korean} (신뢰도: {clf_result.confidence:.1%})")
        print("  Top 3 예측:")
        for rank, (name, prob) in enumerate(clf_result.top3, 1):
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            print(f"    {rank}. {name:14s} [{bar}] {prob:.1%}")

    print(f"\n{'=' * 70}")
    print("  ※ 본 결과는 AI 기반 스크리닝이며, 확진을 위해 전문의 상담이 필요합니다.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
