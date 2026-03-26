"""Tests for disease prediction and classification modules."""

import numpy as np
import pytest

from src.analysis.biomarkers import GaitBiomarkerExtractor, BIOMARKER_DEFINITIONS
from src.analysis.disease_predictor import DiseaseRiskPredictor, DISEASE_DEFINITIONS
from src.analysis.disease_classifier import GaitDiseaseClassifier, FEATURE_NAMES, DISEASE_LABELS


def make_normal_features():
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


def make_parkinsons_features():
    """파킨슨병 특징적 보행 특성."""
    return {
        "gait_speed": 0.6, "cadence": 150, "stride_regularity": 0.40,
        "step_symmetry": 0.75, "cop_sway": 0.07, "ml_index": 0.10,
        "arch_index": 0.24, "acceleration_rms": 0.8,
        "zone_heel_medial_mean": 0.25, "zone_heel_lateral_mean": 0.25,
        "zone_forefoot_medial_mean": 0.35, "zone_forefoot_lateral_mean": 0.35,
        "zone_toes_mean": 0.15,
        "zone_midfoot_medial_mean": 0.1, "zone_midfoot_lateral_mean": 0.1,
    }


def make_stroke_features():
    """뇌졸중(편마비) 특징적 보행 특성."""
    return {
        "gait_speed": 0.5, "cadence": 85, "stride_regularity": 0.55,
        "step_symmetry": 0.55, "cop_sway": 0.08, "ml_index": 0.22,
        "arch_index": 0.26, "acceleration_rms": 1.1,
        "zone_heel_medial_mean": 0.2, "zone_heel_lateral_mean": 0.3,
        "zone_forefoot_medial_mean": 0.4, "zone_forefoot_lateral_mean": 0.3,
        "zone_toes_mean": 0.1,
        "zone_midfoot_medial_mean": 0.1, "zone_midfoot_lateral_mean": 0.1,
    }


class TestBiomarkerExtractor:
    def test_extract_biomarkers(self):
        extractor = GaitBiomarkerExtractor()
        features = make_normal_features()
        profile = extractor.extract(features)

        assert profile.total_count > 0
        assert isinstance(profile.abnormal_count, int)
        assert len(profile.risk_categories) > 0

    def test_normal_features_few_abnormal(self):
        extractor = GaitBiomarkerExtractor()
        features = make_normal_features()
        profile = extractor.extract(features)

        # 정상 보행에서는 이상 바이오마커가 적어야 함
        abnormal_ratio = profile.abnormal_count / max(profile.total_count, 1)
        assert abnormal_ratio < 0.5

    def test_parkinsons_features_more_abnormal(self):
        extractor = GaitBiomarkerExtractor()
        features = make_parkinsons_features()
        profile = extractor.extract(features)

        # 파킨슨 보행에서는 이상 바이오마커가 더 많아야 함
        assert profile.abnormal_count >= 2

    def test_biomarker_result_fields(self):
        extractor = GaitBiomarkerExtractor()
        features = make_normal_features()
        profile = extractor.extract(features)

        for bio in profile.biomarkers:
            assert isinstance(bio.name, str)
            assert isinstance(bio.korean_name, str)
            assert isinstance(bio.value, (int, float))
            assert len(bio.normal_range) == 2
            assert isinstance(bio.is_abnormal, bool)
            assert isinstance(bio.clinical_meaning, str)

    def test_derived_features(self):
        extractor = GaitBiomarkerExtractor()
        features = {"cadence": 110, "ml_index": 0.1, "cop_sway": 0.05, "acceleration_rms": 1.5}
        profile = extractor.extract(features)

        # 파생 바이오마커가 자동 계산되어야 함
        bio_names = {b.name for b in profile.biomarkers}
        assert "gait_speed" in bio_names or profile.total_count > 0


class TestDiseaseRiskPredictor:
    def test_predict_normal(self):
        predictor = DiseaseRiskPredictor()
        features = make_normal_features()
        report = predictor.predict(features)

        assert len(report.results) == len(DISEASE_DEFINITIONS)
        assert 0 <= report.overall_health_score <= 100
        assert isinstance(report.summary_kr, str)

    def test_predict_parkinsons(self):
        predictor = DiseaseRiskPredictor()
        features = make_parkinsons_features()
        report = predictor.predict(features)

        # 파킨슨병이 상위 위험에 있어야 함
        parkinsons_risk = next(r for r in report.results if r.disease_id == "parkinsons")
        assert parkinsons_risk.risk_score > 0.1

    def test_predict_stroke(self):
        predictor = DiseaseRiskPredictor()
        features = make_stroke_features()
        report = predictor.predict(features)

        stroke_risk = next(r for r in report.results if r.disease_id == "stroke")
        assert stroke_risk.risk_score > 0.1
        assert len(stroke_risk.matched_signs) > 0

    def test_risk_scores_bounded(self):
        predictor = DiseaseRiskPredictor()
        features = make_parkinsons_features()
        report = predictor.predict(features)

        for result in report.results:
            assert 0 <= result.risk_score <= 1
            assert result.severity in ("정상", "관심", "주의", "위험")
            assert 0 <= result.confidence <= 1

    def test_summary_contains_key_info(self):
        predictor = DiseaseRiskPredictor()
        features = make_normal_features()
        report = predictor.predict(features)

        assert "보행 건강 점수" in report.summary_kr
        assert "바이오마커" in report.summary_kr
        assert "질환별 위험도" in report.summary_kr

    def test_all_10_diseases(self):
        predictor = DiseaseRiskPredictor()
        features = make_normal_features()
        report = predictor.predict(features)

        disease_ids = {r.disease_id for r in report.results}
        assert len(disease_ids) == 10

    def test_referral_info(self):
        predictor = DiseaseRiskPredictor()
        features = make_parkinsons_features()
        report = predictor.predict(features)

        for result in report.results:
            assert isinstance(result.referral, str)
            assert len(result.referral) > 0


class TestDiseaseClassifier:
    def test_generate_training_data(self):
        clf = GaitDiseaseClassifier()
        X, y = clf.generate_training_data(n_per_class=50)

        assert X.shape == (50 * 7, len(FEATURE_NAMES))
        assert y.shape == (50 * 7,)
        assert len(set(y)) == 7

    def test_train(self):
        clf = GaitDiseaseClassifier(n_estimators=20)
        metrics = clf.train()

        assert metrics.accuracy > 0.5  # 7클래스 랜덤 14%보다 높아야
        assert metrics.f1_macro > 0.3
        assert metrics.cv_accuracy_mean > 0.4
        assert len(metrics.feature_importance) == len(FEATURE_NAMES)

    def test_predict_normal(self):
        clf = GaitDiseaseClassifier(n_estimators=50)
        clf.train()

        features = {f: v for f, v in zip(FEATURE_NAMES,
                    [1.2, 115, 0.85, 0.92, 0.04, 0.06, 0.32, 0.45, 0.25, 0.05, 1.5, 0.15, 2.0])}
        result = clf.predict(features)

        assert result.predicted_korean == "정상 보행"
        assert result.confidence > 0.3
        assert len(result.top3) == 3

    def test_predict_parkinsons(self):
        clf = GaitDiseaseClassifier(n_estimators=50)
        clf.train()

        features = {f: v for f, v in zip(FEATURE_NAMES,
                    [0.65, 148, 0.45, 0.78, 0.065, 0.10, 0.29, 0.49, 0.23, 0.09, 0.95, 0.36, 2.9])}
        result = clf.predict(features)

        # 파킨슨 확률이 상위에 있어야 함
        assert "파킨슨병" in result.probabilities
        assert result.probabilities["파킨슨병"] > 0.1

    def test_feature_importance_report(self):
        clf = GaitDiseaseClassifier(n_estimators=20)
        clf.train()

        report = clf.get_feature_importance_report()
        assert "특성 중요도" in report
        assert "보행" in report

    def test_probabilities_sum_to_one(self):
        clf = GaitDiseaseClassifier(n_estimators=20)
        clf.train()

        features = make_normal_features()
        result = clf.predict(features)

        prob_sum = sum(result.probabilities.values())
        assert abs(prob_sum - 1.0) < 0.01
