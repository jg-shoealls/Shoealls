"""구조적 로깅 설정 및 요청 추적 미들웨어.

로그 형식: JSON (프로덕션) | 컬러 텍스트 (개발)
요청마다 고유 request_id가 할당되어 전체 추적 가능.
"""

import logging
import json
import time
import uuid
import os
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")  # "json" | "text"


class JSONFormatter(logging.Formatter):
    """구조적 JSON 로그 포매터."""

    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        # extra 필드 병합
        for key in ("request_id", "method", "path", "status", "duration_ms", "api_key_id"):
            if hasattr(record, key):
                log[key] = getattr(record, key)
        return json.dumps(log, ensure_ascii=False)


class ColorTextFormatter(logging.Formatter):
    """개발용 컬러 텍스트 포매터."""

    COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = self.formatTime(record, "%H:%M:%S")
        msg = record.getMessage()
        extras = []
        for key in ("request_id", "method", "path", "status", "duration_ms"):
            if hasattr(record, key):
                extras.append(f"{key}={getattr(record, key)}")
        extra_str = "  " + " ".join(extras) if extras else ""
        return f"{ts} {color}{record.levelname:8s}{self.RESET} {record.name}: {msg}{extra_str}"


def setup_logging() -> logging.Logger:
    """애플리케이션 로깅 초기화."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # 기존 핸들러 제거 (uvicorn 재설정 방지)
    root.handlers.clear()

    handler = logging.StreamHandler()
    if LOG_FORMAT == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ColorTextFormatter())

    root.addHandler(handler)

    # uvicorn 로거도 동일 포매터 적용
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False

    return logging.getLogger("shoealls")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """모든 HTTP 요청에 request_id 부여 후 구조적 로그 출력."""

    def __init__(self, app, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()

        extra = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        }
        self.logger.info("→ request", extra=extra)

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            self.logger.error(
                "request failed",
                exc_info=exc,
                extra={**extra, "duration_ms": duration_ms},
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        status = response.status_code
        level = logging.WARNING if status >= 400 else logging.INFO
        self.logger.log(
            level,
            "← response",
            extra={**extra, "status": status, "duration_ms": duration_ms},
        )

        response.headers["X-Request-ID"] = request_id
        return response
