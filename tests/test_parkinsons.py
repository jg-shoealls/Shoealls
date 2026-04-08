"""Tests for the Parkinson's-specific analysis engine and API endpoint."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.analysis.parkinsons_analyzer import (
    ParkinsonsAnalyzer,
    ParkinsonsReport,
    PARKINSONS_SUB_PATTERNS,
    HOEHN_YAHR_STAGES,
)
from src.api.server import app

client = TestClient(app)
analyzer = ParkinsonsAnalyzer()


# ── Feature fixtures ─────────────────────────────────────────────────

NORMAL_FEATURES = {
    "gait_speed": 1.25,
    "cadence": 118,
    "stride_regularity": 0.88,
    "step_symmetry": 0.93,
    "cop_sway": 0.035,
    "ml_index": 0.03,
    "arch_index": 0.24,
    "acceleration_rms": 1.6,
    "zone_heel_medial_mean": 0.32,
    "zone_heel_lateral_mean": 0.30,
    "zone_forefoot_medial_mean": 0.33,
    "zone_forefoot_lateral_mean": 0.30,
    "zone_toes_mean": 0.18,
    "zone_midfoot_medial_mean": 0.10,
    "zone_midfoot_lateral_mean": 0.08,
}

EARLY_PARKINSONS = {
    "gait_speed": 0.85,
    "cadence": 135,
    "stride_regularity": 0.62,
    "step_symmetry": 0.82,
    "cop_sway": 0.055,
    "ml_index": 0.06,
    "arch_index": 0.25,
    "acceleration_rms": 1.0,
    "zone_heel_medial_mean": 0.28,
    "zone_heel_lateral_mean": 0.26,
    "zone_forefoot_medial_mean": 0.35,
    "zone_forefoot_lateral_mean": 0.32,
    "zone_toes_mean": 0.14,
    "zone_midfoot_medial_mean": 0.10,
    "zone_midfoot_lateral_mean": 0.09,
}

MODERATE_PARKINSONS = {
    "gait_speed": 0.60,
    "cadence": 155,
    "stride_regularity": 0.38,
    "step_symmetry": 0.70,
    "cop_sway": 0.10,
    "ml_index": 0.12,
    "arch_index": 0.22,
    "acceleration_rms": 0.5,
    "zone_heel_medial_mean": 0.22,
    "zone_heel_lateral_mean": 0.20,
    "zone_forefoot_medial_mean": 0.40,
    "zone_forefoot_lateral_mean": 0.38,
    "zone_toes_mean": 0.10,
    "zone_midfoot_medial_mean": 0.08,
    "zone_midfoot_lateral_mean": 0.08,
}

SEVERE_PARKINSONS = {
    "gait_speed": 0.35,
    "cadence": 170,
    "stride_regularity": 0.25,
    "step_symmetry": 0.55,
    "cop_sway": 0.16,
    "ml_index": 0.18,
    "arch_index": 0.20,
    "acceleration_rms": 0.25,
    "zone_heel_medial_mean": 0.15,
    "zone_heel_lateral_mean": 0.14,
    "zone_forefoot_medial_mean": 0.45,
    "zone_forefoot_lateral_mean": 0.42,
    "zone_toes_mean": 0.08,
    "zone_midfoot_medial_mean": 0.07,
    "zone_midfoot_lateral_mean": 0.06,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unit tests: ParkinsonsAnalyzer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestParkinsonsAnalyzer:
    def test_normal_low_risk(self):
        report = analyzer.analyze(NORMAL_FEATURES)
        assert report.risk_score < 0.15
        assert report.hoehn_yahr_stage <= 1
        assert len(report.detected_patterns) == 0
        assert report.risk_label == "정상"

    def test_early_parkinsons_detected(self):
        report = analyzer.analyze(EARLY_PARKINSONS)
        assert report.risk_score > 0.10
        assert len(report.detected_patterns) > 0

    def test_moderate_higher_risk(self):
        report = analyzer.analyze(MODERATE_PARKINSONS)
        assert report.risk_score > analyzer.analyze(EARLY_PARKINSONS).risk_score
        assert report.hoehn_yahr_stage >= 2

    def test_severe_highest_risk(self):
        report = analyzer.analyze(SEVERE_PARKINSONS)
        assert report.risk_score > analyzer.analyze(MODERATE_PARKINSONS).risk_score
        assert report.hoehn_yahr_stage >= 3
        assert report.risk_label in ("경고", "위험")

    def test_all_sub_patterns_evaluated(self):
        report = analyzer.analyze(MODERATE_PARKINSONS)
        assert len(report.sub_patterns) == len(PARKINSONS_SUB_PATTERNS)
        pattern_ids = {p.pattern_id for p in report.sub_patterns}
        assert pattern_ids == set(PARKINSONS_SUB_PATTERNS.keys())

    def test_sub_pattern_scores_bounded(self):
        report = analyzer.analyze(SEVERE_PARKINSONS)
        for p in report.sub_patterns:
            assert 0.0 <= p.score <= 1.0

    def test_hoehn_yahr_stages_valid(self):
        report = analyzer.analyze(SEVERE_PARKINSONS)
        assert 0 <= report.hoehn_yahr_stage <= 5
        assert report.hoehn_yahr_label != ""
        assert report.hoehn_yahr_description != ""

    def test_key_findings_nonempty_for_parkinsons(self):
        report = analyzer.analyze(MODERATE_PARKINSONS)
        assert len(report.key_findings) > 0

    def test_recommendations_nonempty_for_parkinsons(self):
        report = analyzer.analyze(MODERATE_PARKINSONS)
        assert len(report.recommendations) > 0

    def test_confidence_bounded(self):
        report = analyzer.analyze(NORMAL_FEATURES)
        assert 0.0 <= report.confidence <= 1.0

    def test_detected_is_subset_of_sub_patterns(self):
        report = analyzer.analyze(MODERATE_PARKINSONS)
        detected_ids = {p.pattern_id for p in report.detected_patterns}
        all_ids = {p.pattern_id for p in report.sub_patterns}
        assert detected_ids.issubset(all_ids)

    def test_indicator_details_present(self):
        report = analyzer.analyze(MODERATE_PARKINSONS)
        for p in report.sub_patterns:
            assert len(p.indicator_details) > 0
            for d in p.indicator_details:
                assert "indicator" in d
                assert "status" in d
                assert "score" in d

    def test_postural_instability_triggers_hy3(self):
        """If postural instability is detected, H&Y should be at least 3."""
        report = analyzer.analyze(SEVERE_PARKINSONS)
        postural = next(
            (p for p in report.detected_patterns if p.pattern_id == "postural_instability"),
            None,
        )
        if postural is not None:
            assert report.hoehn_yahr_stage >= 3

    def test_risk_monotonic_with_severity(self):
        """Risk should increase: normal < early < moderate < severe."""
        scores = [
            analyzer.analyze(NORMAL_FEATURES).risk_score,
            analyzer.analyze(EARLY_PARKINSONS).risk_score,
            analyzer.analyze(MODERATE_PARKINSONS).risk_score,
            analyzer.analyze(SEVERE_PARKINSONS).risk_score,
        ]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], f"scores[{i}]={scores[i]} >= scores[{i+1}]={scores[i+1]}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API endpoint tests: /parkinsons
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestParkinsonsAPI:
    def test_parkinsons_normal(self):
        resp = client.post("/parkinsons", json={"features": NORMAL_FEATURES})
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_score"] < 0.15
        assert data["hoehn_yahr_stage"] <= 1
        assert len(data["detected_patterns"]) == 0

    def test_parkinsons_moderate(self):
        resp = client.post("/parkinsons", json={"features": MODERATE_PARKINSONS})
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_score"] > 0.2
        assert data["hoehn_yahr_stage"] >= 2
        assert len(data["detected_patterns"]) > 0

    def test_parkinsons_severe(self):
        resp = client.post("/parkinsons", json={"features": SEVERE_PARKINSONS})
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_score"] > data["confidence"] * 0.3
        assert data["hoehn_yahr_stage"] >= 3
        assert len(data["recommendations"]) > 0
        assert len(data["key_findings"]) > 0

    def test_parkinsons_response_structure(self):
        resp = client.post("/parkinsons", json={"features": MODERATE_PARKINSONS})
        data = resp.json()
        assert "risk_score" in data
        assert "risk_label" in data
        assert "hoehn_yahr_stage" in data
        assert "hoehn_yahr_label" in data
        assert "hoehn_yahr_description" in data
        assert "sub_patterns" in data
        assert "detected_patterns" in data
        assert "key_findings" in data
        assert "recommendations" in data
        assert "confidence" in data

    def test_parkinsons_sub_pattern_structure(self):
        resp = client.post("/parkinsons", json={"features": MODERATE_PARKINSONS})
        data = resp.json()
        for p in data["sub_patterns"]:
            assert "pattern_id" in p
            assert "korean_name" in p
            assert "score" in p
            assert "detected" in p
            assert "description" in p
            assert "clinical_meaning" in p
            assert "indicator_details" in p

    def test_parkinsons_with_sensor_data(self):
        rng = np.random.RandomState(42)
        T, H, W = 128, 16, 8
        pressure = rng.rand(T, 1, H, W).astype(np.float32) * 0.3
        pressure[:, 0, 11:16, :] += 0.4
        pressure[:, 0, 3:7, :] += 0.35
        imu = rng.randn(6, T).astype(np.float32)
        t = np.linspace(0, 4 * np.pi, T)
        imu[1] += np.sin(t) * 2

        resp = client.post("/parkinsons", json={
            "sensor_data": {
                "imu": imu.tolist(),
                "pressure": pressure.tolist(),
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["risk_score"] <= 1
        assert 0 <= data["hoehn_yahr_stage"] <= 5

    def test_parkinsons_no_input_422(self):
        resp = client.post("/parkinsons", json={})
        assert resp.status_code == 422

    def test_parkinsons_findings_contain_korean(self):
        resp = client.post("/parkinsons", json={"features": MODERATE_PARKINSONS})
        data = resp.json()
        for finding in data["key_findings"]:
            assert isinstance(finding, str)
            assert len(finding) > 0

    def test_parkinsons_recommendations_for_severe(self):
        resp = client.post("/parkinsons", json={"features": SEVERE_PARKINSONS})
        data = resp.json()
        recs_text = " ".join(data["recommendations"])
        assert "신경과" in recs_text
