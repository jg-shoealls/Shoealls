"""API 통합 테스트 — 서비스 레이어 직접 호출 (httpx 불필요).

FastAPI TestClient를 사용하여 실제 HTTP 레이어까지 검증합니다.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app
from api.examples import (
    generate_sample_sensor_data,
    NORMAL_GAIT_FEATURES,
    PARKINSONS_GAIT_FEATURES,
)


@pytest.fixture(scope="module")
def client():
    """TestClient는 lifespan(warmup)까지 실행합니다."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def normal_sensor():
    return generate_sample_sensor_data(gait_class=0)


@pytest.fixture(scope="module")
def parkinsons_sensor():
    return generate_sample_sensor_data(gait_class=3)


# ── Health ─────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["version"] == "0.1.0"


# ── Sample Endpoint ────────────────────────────────────────────────────

class TestSample:
    def test_sample_normal(self, client):
        r = client.get("/api/v1/sample", params={"gait_profile": "normal"})
        assert r.status_code == 200
        body = r.json()
        assert body["gait_profile"] == "normal"
        assert "sensor_data" in body
        assert "features" in body
        sd = body["sensor_data"]
        assert len(sd["imu"]) == 128
        assert len(sd["imu"][0]) == 6
        assert len(sd["pressure"]) == 16
        assert len(sd["pressure"][0]) == 8
        assert len(sd["skeleton"]) == 128
        assert len(sd["skeleton"][0]) == 17
        assert len(sd["skeleton"][0][0]) == 3

    @pytest.mark.parametrize("profile", ["normal", "parkinsons", "stroke", "fall_risk"])
    def test_sample_all_profiles(self, client, profile):
        r = client.get("/api/v1/sample", params={"gait_profile": profile})
        assert r.status_code == 200

    def test_sample_invalid_profile(self, client):
        r = client.get("/api/v1/sample", params={"gait_profile": "invalid"})
        assert r.status_code == 400


# ── Gait Classification ────────────────────────────────────────────────

class TestClassify:
    def test_classify_returns_valid_class(self, client, normal_sensor):
        r = client.post("/api/v1/classify", json={"sensor_data": normal_sensor})
        assert r.status_code == 200
        body = r.json()
        assert body["prediction"] in {"normal", "antalgic", "ataxic", "parkinsonian"}
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["is_demo_mode"] is True  # 체크포인트 없음
        probs = body["class_probabilities"]
        assert set(probs.keys()) == {"normal", "antalgic", "ataxic", "parkinsonian"}
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_classify_probabilities_sum_to_one(self, client, parkinsons_sensor):
        r = client.post("/api/v1/classify", json={"sensor_data": parkinsons_sensor})
        assert r.status_code == 200
        probs = r.json()["class_probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_classify_missing_imu_field(self, client):
        r = client.post("/api/v1/classify", json={
            "sensor_data": {"pressure": [[0.1] * 8] * 16, "skeleton": []}
        })
        assert r.status_code == 422  # Validation error


# ── Disease Risk ───────────────────────────────────────────────────────

class TestDiseaseRisk:
    def test_normal_no_high_risk(self, client):
        r = client.post("/api/v1/disease-risk", json={"features": NORMAL_GAIT_FEATURES})
        assert r.status_code == 200
        body = r.json()
        assert "ml_prediction" in body
        assert 0.0 <= body["ml_confidence"] <= 1.0
        # 정상 보행 → top_diseases 없거나 낮은 위험
        for d in body["top_diseases"]:
            assert d["risk_score"] < 0.75, "정상 보행인데 위험도가 너무 높음"

    def test_parkinsons_detected(self, client):
        r = client.post("/api/v1/disease-risk", json={"features": PARKINSONS_GAIT_FEATURES})
        assert r.status_code == 200
        body = r.json()
        # ML이 파킨슨을 최상위로 예측해야 함
        assert "파킨슨" in body["ml_prediction_kr"] or "parkinsons" in body["ml_prediction"]

    def test_response_structure(self, client):
        r = client.post("/api/v1/disease-risk", json={"features": NORMAL_GAIT_FEATURES})
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["top_diseases"], list)
        assert isinstance(body["ml_top3"], list)
        assert isinstance(body["abnormal_biomarkers"], list)
        assert len(body["ml_top3"]) == 3
        for item in body["ml_top3"]:
            assert "name_kr" in item
            assert "probability" in item


# ── Injury Risk ────────────────────────────────────────────────────────

class TestInjuryRisk:
    def test_returns_valid_structure(self, client):
        r = client.post("/api/v1/injury-risk", json={"features": NORMAL_GAIT_FEATURES})
        assert r.status_code == 200
        body = r.json()
        assert "predicted_injury" in body
        assert "predicted_injury_kr" in body
        assert 0.0 <= body["confidence"] <= 1.0
        assert 0.0 <= body["combined_risk_score"] <= 1.0
        assert body["combined_risk_grade"] in {"정상", "경미", "주의", "경고", "위험"}
        assert isinstance(body["body_risk_map"], dict)
        assert isinstance(body["priority_actions"], list)
        assert isinstance(body["top3"], list)
        assert len(body["top3"]) == 3

    def test_timeline_present(self, client):
        r = client.post("/api/v1/injury-risk", json={"features": NORMAL_GAIT_FEATURES})
        assert r.status_code == 200
        assert r.json()["timeline"]  # 비어있지 않아야 함


# ── Chain-of-Reasoning ────────────────────────────────────────────────

class TestReasoning:
    def test_reasoning_structure(self, client, normal_sensor):
        r = client.post("/api/v1/reasoning", json={"sensor_data": normal_sensor})
        assert r.status_code == 200
        body = r.json()
        assert body["final_prediction"] in {"normal", "antalgic", "ataxic", "parkinsonian"}
        assert 0.0 <= body["confidence"] <= 1.0
        assert 0.0 <= body["uncertainty"] <= 1.0
        assert 0.0 <= body["evidence_strength"] <= 1.0
        assert body["is_demo_mode"] is True

    def test_reasoning_trace_steps(self, client, normal_sensor):
        r = client.post("/api/v1/reasoning", json={"sensor_data": normal_sensor})
        body = r.json()
        trace = body["reasoning_trace"]
        assert len(trace) >= 2  # 최소 초기가설 + 1단계
        for step in trace:
            assert "step" in step
            assert "label" in step
            assert "prediction_kr" in step
            assert 0.0 <= step["probability"] <= 1.0

    def test_anomaly_findings_three_modalities(self, client, normal_sensor):
        r = client.post("/api/v1/reasoning", json={"sensor_data": normal_sensor})
        body = r.json()
        assert len(body["anomaly_findings"]) == 3
        modalities = {f["modality"] for f in body["anomaly_findings"]}
        assert "IMU (관성센서)" in modalities
        assert "족저압 센서" in modalities
        assert "스켈레톤" in modalities

    def test_report_kr_present(self, client, normal_sensor):
        r = client.post("/api/v1/reasoning", json={"sensor_data": normal_sensor})
        body = r.json()
        assert len(body["report_kr"]) > 100  # 최소한의 리포트 길이


# ── Full Analysis ─────────────────────────────────────────────────────

class TestAnalyze:
    def test_full_analysis_structure(self, client, normal_sensor):
        r = client.post("/api/v1/analyze", json={
            "sensor_data": normal_sensor,
            "features": NORMAL_GAIT_FEATURES,
        })
        assert r.status_code == 200
        body = r.json()
        assert "classify" in body
        assert "disease_risk" in body
        assert "injury_risk" in body
        assert "reasoning" in body

    def test_full_analysis_consistency(self, client, normal_sensor):
        """classify와 analyze의 분류 결과가 동일해야 함 (같은 입력)."""
        r_classify = client.post("/api/v1/classify", json={"sensor_data": normal_sensor})
        r_analyze = client.post("/api/v1/analyze", json={
            "sensor_data": normal_sensor,
            "features": NORMAL_GAIT_FEATURES,
        })
        assert r_classify.status_code == 200
        assert r_analyze.status_code == 200
        assert r_classify.json()["prediction"] == r_analyze.json()["classify"]["prediction"]
