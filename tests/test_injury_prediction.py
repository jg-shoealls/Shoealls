"""Tests for abnormal gait pattern detection and injury risk prediction."""

import numpy as np
import pytest

from src.analysis.gait_anomaly import (
    GaitAnomalyDetector, GaitAnomalyReport, AnomalyPattern,
    ANOMALY_DEFINITIONS, INJURY_CATEGORIES,
)
from src.analysis.injury_predictor import (
    InjuryRiskPredictor, InjuryPrediction, ComprehensiveInjuryReport,
    INJURY_LABELS, PREDICTOR_FEATURES,
)


def make_normal_gait():
    """정상 보행 특성."""
    return {
        "gait_speed": 1.2, "cadence": 115, "stride_regularity": 0.85,
        "step_symmetry": 0.92, "cop_sway": 0.04, "ml_index": 0.03,
        "arch_index": 0.25, "acceleration_rms": 1.5,
        "zone_heel_medial_mean": 0.3, "zone_heel_lateral_mean": 0.3,
        "zone_forefoot_medial_mean": 0.35, "zone_forefoot_lateral_mean": 0.3,
        "zone_toes_mean": 0.15,
        "zone_midfoot_medial_mean": 0.1, "zone_midfoot_lateral_mean": 0.08,
    }


def make_antalgic_gait():
    """절뚝거림(통증 회피) 보행 특성."""
    return {
        "gait_speed": 0.6, "cadence": 85, "stride_regularity": 0.55,
        "step_symmetry": 0.55, "cop_sway": 0.08, "ml_index": 0.18,
        "arch_index": 0.26, "acceleration_rms": 0.85,
        "zone_heel_medial_mean": 0.18, "zone_heel_lateral_mean": 0.30,
        "zone_forefoot_medial_mean": 0.42, "zone_forefoot_lateral_mean": 0.30,
        "zone_toes_mean": 0.10,
        "zone_midfoot_medial_mean": 0.08, "zone_midfoot_lateral_mean": 0.12,
    }


def make_forefoot_overload_gait():
    """전족부 과부하 보행 특성."""
    return {
        "gait_speed": 1.1, "cadence": 120, "stride_regularity": 0.80,
        "step_symmetry": 0.88, "cop_sway": 0.05, "ml_index": 0.04,
        "arch_index": 0.22, "acceleration_rms": 1.6,
        "zone_heel_medial_mean": 0.08, "zone_heel_lateral_mean": 0.06,
        "zone_forefoot_medial_mean": 0.50, "zone_forefoot_lateral_mean": 0.45,
        "zone_toes_mean": 0.25,
        "zone_midfoot_medial_mean": 0.06, "zone_midfoot_lateral_mean": 0.05,
    }


def make_fall_risk_gait():
    """낙상 고위험 보행 특성 (고령/균형 장애)."""
    return {
        "gait_speed": 0.55, "cadence": 80, "stride_regularity": 0.42,
        "step_symmetry": 0.70, "cop_sway": 0.12, "ml_index": 0.16,
        "arch_index": 0.30, "acceleration_rms": 0.7,
        "zone_heel_medial_mean": 0.25, "zone_heel_lateral_mean": 0.28,
        "zone_forefoot_medial_mean": 0.35, "zone_forefoot_lateral_mean": 0.32,
        "zone_toes_mean": 0.10,
        "zone_midfoot_medial_mean": 0.10, "zone_midfoot_lateral_mean": 0.12,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GaitAnomalyDetector 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestGaitAnomalyDetector:
    def test_detect_normal_gait(self):
        detector = GaitAnomalyDetector()
        report = detector.detect(make_normal_gait())

        assert isinstance(report, GaitAnomalyReport)
        assert len(report.patterns) == len(ANOMALY_DEFINITIONS)
        assert report.anomaly_score < 0.3
        assert len(report.abnormal_patterns) <= 3

    def test_detect_antalgic_gait(self):
        detector = GaitAnomalyDetector()
        report = detector.detect(make_antalgic_gait())

        assert report.anomaly_score > 0.2
        assert len(report.abnormal_patterns) >= 3

        # 절뚝거림 패턴이 감지되어야 함
        pattern_ids = {p.pattern_id for p in report.abnormal_patterns}
        assert "antalgic_gait" in pattern_ids

    def test_detect_forefoot_overload(self):
        detector = GaitAnomalyDetector()
        report = detector.detect(make_forefoot_overload_gait())

        pattern_ids = {p.pattern_id for p in report.abnormal_patterns}
        assert "forefoot_overload" in pattern_ids

    def test_detect_fall_risk(self):
        detector = GaitAnomalyDetector()
        report = detector.detect(make_fall_risk_gait())

        assert report.anomaly_score > 0.3

        # 체중심 불안정, 보행 속도 저하 등이 감지되어야 함
        pattern_ids = {p.pattern_id for p in report.abnormal_patterns}
        assert "cop_instability" in pattern_ids
        assert "slow_gait" in pattern_ids

    def test_anomaly_score_bounded(self):
        detector = GaitAnomalyDetector()
        for features in [make_normal_gait(), make_antalgic_gait(), make_fall_risk_gait()]:
            report = detector.detect(features)
            assert 0 <= report.anomaly_score <= 1

    def test_severity_labels(self):
        detector = GaitAnomalyDetector()
        report = detector.detect(make_antalgic_gait())

        valid_labels = {"정상", "경미", "주의", "경고", "위험"}
        for p in report.patterns:
            assert p.severity_label in valid_labels
        assert report.anomaly_grade in valid_labels

    def test_pattern_fields(self):
        detector = GaitAnomalyDetector()
        report = detector.detect(make_antalgic_gait())

        for p in report.patterns:
            assert isinstance(p.pattern_id, str)
            assert isinstance(p.korean_name, str)
            assert isinstance(p.severity, float)
            assert isinstance(p.description, str)
            assert isinstance(p.injury_risks, list)

    def test_injury_risk_summary(self):
        detector = GaitAnomalyDetector()
        report = detector.detect(make_antalgic_gait())

        # 비정상 패턴이 있으면 부상 위험도 있어야 함
        assert len(report.injury_risk_summary) > 0
        for injury_name, score in report.injury_risk_summary.items():
            assert 0 <= score <= 1
            assert isinstance(injury_name, str)

    def test_report_contains_korean(self):
        detector = GaitAnomalyDetector()
        report = detector.detect(make_antalgic_gait())

        assert "비정상 보행 패턴" in report.summary_kr
        assert "부상 위험" in report.summary_kr

    def test_all_12_patterns_checked(self):
        detector = GaitAnomalyDetector()
        report = detector.detect(make_normal_gait())

        assert len(report.patterns) == 12

    def test_correction_for_abnormal(self):
        detector = GaitAnomalyDetector()
        report = detector.detect(make_antalgic_gait())

        for p in report.abnormal_patterns:
            assert len(p.correction) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# InjuryRiskPredictor 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestInjuryRiskPredictor:
    def test_generate_training_data(self):
        predictor = InjuryRiskPredictor()
        X, y = predictor.generate_training_data(n_per_class=50)

        assert X.shape == (50 * 9, len(PREDICTOR_FEATURES))
        assert y.shape == (50 * 9,)
        assert len(set(y)) == 9

    def test_train(self):
        predictor = InjuryRiskPredictor(n_estimators=30)
        metrics = predictor.train()

        assert metrics.accuracy > 0.5
        assert metrics.f1_macro > 0.3
        assert metrics.cv_accuracy_mean > 0.4
        assert len(metrics.feature_importance) == len(PREDICTOR_FEATURES)

    def test_predict_normal(self):
        predictor = InjuryRiskPredictor(n_estimators=50)
        predictor.train()

        result = predictor.predict(make_normal_gait())

        assert result.predicted_korean == "낮은 위험"
        assert result.confidence > 0.3
        assert len(result.top3) == 3

    def test_predict_fall_risk(self):
        predictor = InjuryRiskPredictor(n_estimators=50)
        predictor.train()

        result = predictor.predict(make_fall_risk_gait())

        # 낙상 위험이 상위에 있어야 함
        assert "낙상 위험" in result.probabilities
        assert result.probabilities["낙상 위험"] > 0.05

    def test_probabilities_sum_to_one(self):
        predictor = InjuryRiskPredictor(n_estimators=30)
        predictor.train()

        result = predictor.predict(make_antalgic_gait())
        prob_sum = sum(result.probabilities.values())
        assert abs(prob_sum - 1.0) < 0.01

    def test_body_risk_map(self):
        predictor = InjuryRiskPredictor(n_estimators=30)
        predictor.train()

        result = predictor.predict(make_antalgic_gait())
        assert isinstance(result.body_risk_map, dict)
        for part, score in result.body_risk_map.items():
            assert 0 <= score <= 1

    def test_timeline(self):
        predictor = InjuryRiskPredictor(n_estimators=30)
        predictor.train()

        result = predictor.predict(make_antalgic_gait())
        assert isinstance(result.timeline, str)
        assert len(result.timeline) > 0

    def test_comprehensive_report(self):
        predictor = InjuryRiskPredictor(n_estimators=50)
        predictor.train()

        report = predictor.predict_comprehensive(make_antalgic_gait())

        assert isinstance(report, ComprehensiveInjuryReport)
        assert isinstance(report.anomaly_report, GaitAnomalyReport)
        assert isinstance(report.ml_prediction, InjuryPrediction)
        assert 0 <= report.combined_risk_score <= 1
        assert report.combined_risk_grade in {"정상", "경미", "주의", "경고", "위험"}
        assert len(report.priority_actions) > 0
        assert "슈올즈 AI" in report.summary_kr

    def test_comprehensive_body_map(self):
        predictor = InjuryRiskPredictor(n_estimators=50)
        predictor.train()

        report = predictor.predict_comprehensive(make_fall_risk_gait())

        assert len(report.body_risk_map) > 0
        for part, score in report.body_risk_map.items():
            assert 0 <= score <= 1

    def test_comprehensive_normal_low_risk(self):
        predictor = InjuryRiskPredictor(n_estimators=50)
        predictor.train()

        report = predictor.predict_comprehensive(make_normal_gait())
        assert report.combined_risk_score < 0.5
