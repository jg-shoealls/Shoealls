"""API 키 인증 미들웨어.

환경변수 API_KEYS (쉼표 구분)로 허용 키 목록을 설정합니다.
  API_KEYS=key-abc123,key-xyz789

헤더:  X-API-Key: key-abc123
쿼리:  ?api_key=key-abc123

API_KEYS 미설정 시 인증 비활성화 (개발/데모 모드).
"""

import os
import time
import hashlib
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from starlette.middleware.base import BaseHTTPMiddleware

# ── 허용 키 로드 ──────────────────────────────────────────────────────
_raw = os.getenv("API_KEYS", "").strip()
_ALLOWED_KEYS: set[str] = (
    {k.strip() for k in _raw.split(",") if k.strip()} if _raw else set()
)
AUTH_ENABLED = bool(_ALLOWED_KEYS)

# 공개 경로 (인증 없이 접근 가능)
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/api/v1/sample"}

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_api_key_query = APIKeyQuery(name="api_key", auto_error=False)


def _key_id(key: str) -> str:
    """로그용 키 앞 8자 마스킹."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


class APIKeyMiddleware(BaseHTTPMiddleware):
    """X-API-Key 헤더 또는 ?api_key 쿼리 파라미터로 인증."""

    async def dispatch(self, request: Request, call_next):
        if not AUTH_ENABLED:
            return await call_next(request)

        # 공개 경로 통과
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # 키 추출 (헤더 우선)
        key: Optional[str] = (
            request.headers.get("X-API-Key")
            or request.query_params.get("api_key")
        )

        if not key or key not in _ALLOWED_KEYS:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": {
                        "error": "Unauthorized",
                        "message": "유효한 API 키가 필요합니다. X-API-Key 헤더 또는 ?api_key 쿼리 파라미터를 사용하세요.",
                    }
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # 인증된 키 ID를 request state에 저장 (로깅용)
        request.state.api_key_id = _key_id(key)
        return await call_next(request)
