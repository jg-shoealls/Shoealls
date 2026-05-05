# Shoealls — 멀티모달 보행 AI 분석 서비스

IMU · 족저압 · 스켈레톤 센서 데이터를 융합해 보행 패턴을 분류하고,
질환 위험도·부상 위험을 예측하는 AI 서비스입니다.

## 핵심 기능

| 기능 | 설명 |
|------|------|
| 보행 패턴 분류 | 4-class (정상/파킨슨/절뚝거림/운동실조) |
| 질환 위험 예측 | 11개 질환 클래스 ML 분류 |
| 부상 위험 평가 | 6가지 부상 유형 족저압 기반 평가 |
| Chain-of-Reasoning | 단계별 추론 트레이스 + 불확실성 정량화 |
| 대시보드 UI | Next.js 기반 실시간 분석 대시보드 |

## 아키텍처

```
IMU [128,6] ──────► IMU Encoder (1D-CNN + LSTM) ──┐
                                                   ├── Cross-Modal Attention ──► Classifier
Pressure [16,8] ──► Pressure Encoder (2D-CNN) ────┤
                                                   │
Skeleton [128,17,3] ► Skeleton Encoder (GCN) ─────┘
```

```
Frontend (Next.js 16)
    ↕ /api proxy
FastAPI (api/)
    ↕ inference
src/analysis/   ← 보행 지표 추출, 질환 분류, 부상 평가
src/models/     ← 멀티모달 딥러닝 모델
outputs/        ← 학습된 체크포인트
```

## 빠른 시작

### 1. 백엔드 (FastAPI)

```bash
pip install -r requirements.txt

# 데모 모드 (체크포인트 없이 실행)
uvicorn api.main:app --reload --port 8000

# 학습 후 실제 모델로 실행
python -m src.training.train --samples 500 --epochs 30 --verify
uvicorn api.main:app --reload --port 8000
```

### 2. 프론트엔드 (Next.js)

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

### 3. Docker Compose

```bash
# 프로덕션
docker compose up api

# 개발 (hot-reload)
docker compose --profile dev up api-dev
```

### 4. API 키 인증 활성화

```bash
# .env 파일 생성
cp .env.example .env
# API_KEYS=key1,key2 로 설정하면 /api/v1/analyze 등에 인증 적용
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 |
| GET | `/api/v1/sample` | 샘플 센서 데이터 생성 |
| POST | `/api/v1/classify` | 보행 패턴 분류 |
| POST | `/api/v1/disease-risk` | 질환 위험도 예측 |
| POST | `/api/v1/injury-risk` | 부상 위험 평가 |
| POST | `/api/v1/reasoning` | Chain-of-Reasoning 추론 |
| POST | `/api/v1/analyze` | 통합 분석 (전체) |

```bash
# 예시: 샘플 생성 후 통합 분석
curl http://localhost:8000/api/v1/sample?gait_profile=parkinsons | \
  jq '{sensor_data: .sensor_data, features: .features}' | \
  curl -X POST http://localhost:8000/api/v1/analyze \
    -H 'Content-Type: application/json' -d @-
```

## 학습 파이프라인

```bash
# 빠른 학습 (테스트용)
python -m src.training.train --samples 80 --epochs 5 --verify

# 전체 학습
python -m src.training.train --samples 500 --epochs 50 --lr 0.001

# 이어서 학습
python -m src.training.train --resume --samples 500 --epochs 100
```

학습 결과는 `outputs/best_model.pt` 및 `outputs/train_result.json`에 저장됩니다.

## 알고리즘 범위

→ [`docs/algorithm_scope.md`](docs/algorithm_scope.md) 참고

- 입력: IMU `[128,6]` · 족저압 `[16,8]` · 스켈레톤 `[128,17,3]` @ 128 Hz
- 보행 지표: 13개 (시공간 4 · 족저압 4 · 균형/IMU 5)
- 보행 분류: 4 classes
- 질환 분류: 11 classes · 45개 질환-특이 바이오마커
- 부상 예측: 6 types

## 테스트

```bash
# 전체 테스트
pytest tests/test_api.py tests/test_auth.py -v

# 커버리지 포함
pytest tests/test_api.py tests/test_auth.py --cov=api --cov-report=term-missing
```

## 프로젝트 구조

```
Shoealls/
├── api/                    # FastAPI 서비스
│   ├── main.py             # 라우터 + 미들웨어
│   ├── service.py          # ML 추론 서비스
│   ├── auth.py             # API 키 인증
│   ├── logging_config.py   # 구조화 로깅
│   └── schemas.py          # Pydantic 스키마
├── src/
│   ├── analysis/           # 보행 분석 모듈 (16개)
│   ├── models/             # 딥러닝 모델
│   └── training/           # 학습 파이프라인
├── frontend/               # Next.js 16 대시보드
│   └── src/app/
│       ├── page.tsx        # 메인 대시보드
│       ├── analysis/       # 보행 패턴 분석
│       ├── disease/        # 질환 위험 예측
│       ├── injury/         # 부상 위험 평가
│       ├── reasoning/      # AI 추론 트레이스
│       └── history/        # 분석 이력
├── configs/                # 설정 파일
├── docs/                   # 문서
├── tests/                  # 테스트
├── Dockerfile              # 멀티스테이지 빌드
└── docker-compose.yml
```
