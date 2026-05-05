# ── 빌드 스테이지 ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# 시스템 의존성 (transformers 빌드에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Python 의존성 캐시 레이어
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── 런타임 스테이지 ──────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# curl: HEALTHCHECK에서 사용
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 빌드 결과물 복사
COPY --from=builder /install /usr/local

# 소스코드 복사
COPY src/       src/
COPY api/       api/
COPY configs/   configs/

# 출력 디렉토리 (체크포인트 마운트 포인트)
RUN mkdir -p outputs

# 비루트 사용자
RUN useradd -m -u 1000 shoealls && chown -R shoealls:shoealls /app
USER shoealls

# 환경변수 기본값
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    WORKERS=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]
