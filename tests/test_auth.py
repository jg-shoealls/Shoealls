"""인증 미들웨어 테스트.

API_KEYS 환경변수를 통한 API 키 인증 동작을 검증합니다.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import pytest
from fastapi.testclient import TestClient
from api.examples import generate_sample_sensor_data


def _classify_body():
    """분류 엔드포인트용 요청 바디 (ClassifyRequest 형식)."""
    d = generate_sample_sensor_data(gait_class=0)
    return {
        "sensor_data": {
            "imu": d["imu"],
            "pressure": d["pressure"],
            "skeleton": d["skeleton"],
        }
    }


def _reload_app(api_keys: str | None = None):
    """환경변수를 적용하고 앱을 재로드해 TestClient를 반환."""
    if api_keys is None:
        os.environ.pop("API_KEYS", None)
    else:
        os.environ["API_KEYS"] = api_keys

    import api.auth as m_auth
    import api.main as m_main
    importlib.reload(m_auth)
    importlib.reload(m_main)
    return TestClient(m_main.app, raise_server_exceptions=False)


# ── 데모 모드(인증 비활성) ─────────────────────────────────────────────
class TestDemoMode:
    """API_KEYS 미설정 시 모든 경로 자유 접근."""

    def setup_method(self):
        self.client = _reload_app(api_keys=None)

    def test_health_accessible(self):
        assert self.client.get("/health").status_code == 200

    def test_classify_accessible_without_key(self):
        r = self.client.post("/api/v1/classify", json=_classify_body())
        assert r.status_code == 200

    def test_sample_accessible_without_key(self):
        assert self.client.get("/api/v1/sample").status_code == 200


# ── 인증 활성 모드 ─────────────────────────────────────────────────────
class TestAuthEnabled:
    """API_KEYS 설정 시 키 없는 요청은 401 반환."""

    def setup_method(self):
        self.client = _reload_app("test-secret-key")

    def teardown_method(self):
        os.environ.pop("API_KEYS", None)

    # 공개 경로는 키 없이 접근 가능
    def test_health_public_no_key(self):
        assert self.client.get("/health").status_code == 200

    def test_docs_public_no_key(self):
        assert self.client.get("/docs").status_code == 200

    def test_sample_public_no_key(self):
        assert self.client.get("/api/v1/sample").status_code == 200

    # 보호 경로: 키 없음 → 401
    def test_classify_requires_key(self):
        r = self.client.post("/api/v1/classify", json=_classify_body())
        assert r.status_code == 401

    def test_analyze_requires_key(self):
        r = self.client.post("/api/v1/analyze", json=_classify_body())
        assert r.status_code == 401

    def test_401_body_structure(self):
        r = self.client.post("/api/v1/classify", json=_classify_body())
        body = r.json()
        assert "detail" in body
        assert "error" in body["detail"]

    # 헤더 키 인증
    def test_valid_header_key(self):
        r = self.client.post(
            "/api/v1/classify",
            json=_classify_body(),
            headers={"X-API-Key": "test-secret-key"},
        )
        assert r.status_code == 200

    def test_invalid_header_key(self):
        r = self.client.post(
            "/api/v1/classify",
            json=_classify_body(),
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401

    # 쿼리 파라미터 키 인증
    def test_valid_query_key(self):
        r = self.client.post(
            "/api/v1/classify?api_key=test-secret-key",
            json=_classify_body(),
        )
        assert r.status_code == 200

    # 헤더가 쿼리보다 우선
    def test_header_takes_priority_over_query(self):
        r = self.client.post(
            "/api/v1/classify?api_key=wrong-key",
            json=_classify_body(),
            headers={"X-API-Key": "test-secret-key"},
        )
        assert r.status_code == 200

    # 여러 유효 키
    def test_multiple_valid_keys(self):
        client = _reload_app("key-a,key-b,key-c")
        for key in ["key-a", "key-b", "key-c"]:
            r = client.post(
                "/api/v1/classify",
                json=_classify_body(),
                headers={"X-API-Key": key},
            )
            assert r.status_code == 200, f"key={key} should be valid"


# ── 응답 헤더 ─────────────────────────────────────────────────────────
class TestResponseHeaders:
    """로깅 미들웨어가 X-Request-ID를 응답 헤더에 포함하는지 확인."""

    def setup_method(self):
        self.client = _reload_app(api_keys=None)

    def test_health_has_request_id(self):
        r = self.client.get("/health")
        assert "x-request-id" in r.headers

    def test_request_id_is_8_chars(self):
        r = self.client.get("/health")
        assert len(r.headers.get("x-request-id", "")) == 8

    def test_different_requests_have_unique_ids(self):
        ids = {self.client.get("/health").headers.get("x-request-id") for _ in range(5)}
        assert len(ids) == 5  # 모두 고유
