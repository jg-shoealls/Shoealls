# 백엔드 배포용 Dockerfile
FROM python:3.11-slim

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 및 설정 복사
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/
# outputs 디렉토리 생성 (모델 레지스트리용)
RUN mkdir -p outputs/registry

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV OLLAMA_MODEL=llama3.2
ENV HOST=0.0.0.0
ENV PORT=8000

# API 서버 실행
EXPOSE 8000
CMD ["python", "-m", "src.serving.api", "--host", "0.0.0.0", "--port", "8000"]
