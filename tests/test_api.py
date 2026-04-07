"""Tests for the FastAPI inference API server."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.server import app

client = TestClient(app)


# ── Test data fixtures ───────────────────────────────────────────────

def _make_sensor_payload(pattern="normal"):
    """Generate sensor data payload matching the API schema."""
    rng = np.random.RandomState(42)
    T, H, W = 128, 16, 8

    pressure = rng.rand(T, 1, H, W).astype(np.float32) * 0.3
    if pattern == "normal":
        pressure[:, 0, 11:16, :] += 0.4
        pressure[:, 0, 3:7, :] += 0.35
        pressure[:, 0, 0:3, :] += 0.2
        pressure[:, 0, 7:11, :] += 0.1
    elif pattern == "abnormal":
        pressure[:, 0, 0:7, :] += 0.7
        pressure[:, 0, 11:16, :] += 0.1

    imu = rng.randn(6, T).astype(np.float32)
    t = np.linspace(0, 4 * np.pi, T)
    imu[1] += np.sin(t) * 2

    return {
        "imu": imu.tolist(),
        "pressure": pressure.tolist(),
    }


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

PARKINSONS_FEATURES = {
    "gait_speed": 0.65,
    "cadence": 148,
    "stride_regularity": 0.42,
    "step_symmetry": 0.78,
    "cop_sway": 0.065,
    "ml_index": 0.08,
    "arch_index": 0.23,
    "acceleration_rms": 0.9,
    "zone_heel_medial_mean": 0.28,
    "zone_heel_lateral_mean": 0.25,
    "zone_forefoot_medial_mean": 0.38,
    "zone_forefoot_lateral_mean": 0.35,
    "zone_toes_mean": 0.12,
    "zone_midfoot_medial_mean": 0.10,
    "zone_midfoot_lateral_mean": 0.10,
}


# ── Health check ─────────────────────────────────────────────────────

def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "shoealls-gait-api"


# ── /predict endpoint ────────────────────────────────────────────────

class TestPredict:
    def test_predict_with_features(self):
        resp = client.post("/predict", json={"features": NORMAL_FEATURES})
        assert resp.status_code == 200
        data = resp.json()
        assert "anomaly_score" in data
        assert 0 <= data["anomaly_score"] <= 1
        assert "anomaly_grade" in data
        assert "abnormal_patterns" in data
        assert "gait_features" in data

    def test_predict_with_sensor_data(self):
        sensor = _make_sensor_payload("normal")
        resp = client.post("/predict", json={"sensor_data": sensor})
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["anomaly_score"] <= 1
        assert isinstance(data["gait_features"], dict)

    def test_predict_abnormal_detects_patterns(self):
        resp = client.post("/predict", json={"features": PARKINSONS_FEATURES})
        assert resp.status_code == 200
        data = resp.json()
        assert data["anomaly_score"] > 0
        assert len(data["abnormal_patterns"]) > 0

    def test_predict_no_input_returns_422(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_predict_pattern_fields(self):
        resp = client.post("/predict", json={"features": PARKINSONS_FEATURES})
        data = resp.json()
        for pattern in data["abnormal_patterns"]:
            assert "pattern_id" in pattern
            assert "korean_name" in pattern
            assert "severity" in pattern
            assert "correction" in pattern


# ── /disease-risk endpoint ───────────────────────────────────────────

class TestDiseaseRisk:
    def test_disease_risk_normal(self):
        resp = client.post("/disease-risk", json={"features": NORMAL_FEATURES})
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_health_score" in data
        assert 0 <= data["overall_health_score"] <= 100
        assert "all_results" in data
        assert len(data["all_results"]) > 0
        assert "biomarkers" in data

    def test_disease_risk_parkinsons(self):
        resp = client.post("/disease-risk", json={"features": PARKINSONS_FEATURES})
        data = resp.json()
        assert data["overall_health_score"] < 100
        # Parkinsons-like features should produce higher risk for parkinsons than normal
        parkinsons_risk = next(
            (r for r in data["all_results"] if r["disease_id"] == "parkinsons"), None
        )
        assert parkinsons_risk is not None
        assert parkinsons_risk["risk_score"] > 0

    def test_disease_risk_item_fields(self):
        resp = client.post("/disease-risk", json={"features": PARKINSONS_FEATURES})
        data = resp.json()
        for risk in data["all_results"]:
            assert "disease_id" in risk
            assert "korean_name" in risk
            assert 0 <= risk["risk_score"] <= 1
            assert "severity" in risk
            assert "referral" in risk

    def test_disease_risk_biomarker_fields(self):
        resp = client.post("/disease-risk", json={"features": NORMAL_FEATURES})
        data = resp.json()
        for bm in data["biomarkers"]:
            assert "name" in bm
            assert "korean_name" in bm
            assert "normal_range" in bm
            assert len(bm["normal_range"]) == 2
            assert "is_abnormal" in bm

    def test_disease_risk_with_sensor_data(self):
        sensor = _make_sensor_payload("normal")
        resp = client.post("/disease-risk", json={"sensor_data": sensor})
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_health_score"] > 0


# ── /feedback endpoint ───────────────────────────────────────────────

class TestFeedback:
    def test_feedback_with_features(self):
        resp = client.post("/feedback", json={"features": NORMAL_FEATURES})
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_status" in data
        assert "encouragement" in data
        assert "feedback_items" in data
        assert "injury_risks" in data

    def test_feedback_with_sensor_data(self):
        sensor = _make_sensor_payload("normal")
        resp = client.post("/feedback", json={"sensor_data": sensor})
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_status"] in ["매우 양호", "양호", "개선 권장", "주의 필요"]

    def test_feedback_abnormal_has_items(self):
        sensor = _make_sensor_payload("abnormal")
        resp = client.post("/feedback", json={"sensor_data": sensor})
        data = resp.json()
        assert len(data["injury_risks"]) > 0

    def test_feedback_item_fields(self):
        resp = client.post("/feedback", json={"features": PARKINSONS_FEATURES})
        data = resp.json()
        for item in data["feedback_items"]:
            assert "category" in item
            assert item["category"] in ["exercise", "footwear", "posture", "medical"]
            assert "priority" in item
            assert "title" in item
            assert "exercises" in item

    def test_feedback_injury_risk_fields(self):
        resp = client.post("/feedback", json={"features": NORMAL_FEATURES})
        data = resp.json()
        for risk in data["injury_risks"]:
            assert "name" in risk
            assert "korean_name" in risk
            assert 0 <= risk["risk_score"] <= 1
            assert "severity" in risk
            assert "recommendation" in risk


# ── Validation errors ────────────────────────────────────────────────

class TestValidation:
    def test_no_input_422(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_invalid_imu_shape(self):
        resp = client.post("/predict", json={
            "sensor_data": {
                "imu": [[1.0, 2.0]],
                "pressure": [[[[0.1] * 8] * 16]],
            }
        })
        assert resp.status_code == 422

    def test_features_missing_field(self):
        incomplete = {"gait_speed": 1.0}
        resp = client.post("/predict", json={"features": incomplete})
        assert resp.status_code == 422
