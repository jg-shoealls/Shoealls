# Shoealls — Gait AI · CLAUDE.md

## 프로젝트 개요

스마트 신발 인솔 센서(IMU + 족저압 + 스켈레톤)로 수집한 보행 데이터를 분석해
보행 패턴 분류 · 질환 위험도 · 부상 예측 · Chain-of-Reasoning 추론을 제공하는 FastAPI 서버.

---

## 빠른 시작

```bash
# API 서버 실행
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Docker (프로덕션)
docker compose up api

# 테스트
python -m pytest tests/ --ignore=tests/test_disease_biomarkers.py -q

# 전처리 파이프라인 검증 (32 케이스)
python scripts/validate_preprocessing.py

# 배포 정적 검증 (Docker 데몬 없이)
python scripts/verify_docker_build.py

# API 통합 테스트 (TestClient)
python scripts/test_api.py
```

---

## 아키텍처

### 모델 — `MultimodalGaitNet` (`src/models/multimodal_gait_net.py`)

```
IMU (T,6)         → IMUEncoder (1D-CNN + BiLSTM)  → (B, embed_dim)
Pressure (T,1,H,W)→ PressureEncoder (2D-CNN)       → (B, embed_dim)
Skeleton (3,T,J)  → SkeletonEncoder (ST-GCN)       → (B, embed_dim)
                       ↓
          CrossModalAttentionFusion (Transformer)
                       ↓
              Classifier → 4클래스
```

**4 보행 클래스** (`src/constants.py`):

| 인덱스 | 영문 | 한글 |
|--------|------|------|
| 0 | normal | 정상 보행 |
| 1 | antalgic | 절뚝거림 |
| 2 | ataxic | 운동실조 |
| 3 | parkinsonian | 파킨슨 |

### API 레이어 — `api/`

| 엔드포인트 | 입력 | 반환 |
|-----------|------|------|
| `GET  /health` | — | 서버 버전·상태 |
| `GET  /api/sample` | `?gait_profile=normal` | 합성 센서 데이터 |
| `POST /api/classify` | `SensorData` | 보행 클래스 + 확률 |
| `POST /api/disease-risk` | `GaitFeatures` | 질환별 위험도 |
| `POST /api/injury-risk` | `GaitFeatures` | 부상 부위별 위험도 |
| `POST /api/reasoning` | `SensorData + GaitFeatures` | Chain-of-Reasoning |
| `POST /api/analyze` | `AnalyzeRequest` | 전체 통합 분석 |

### 분석 레이어 — `src/analysis/`

```
common.py          ← EPSILON, severity_label(), linear_risk_score(), compute_derived_features()
disease_predictor.py  규칙기반 질환 위험도 (DISEASE_DEFINITIONS 딕셔너리)
injury_predictor.py   ML 기반 부상 예측 (BaseGaitClassifier 상속)
injury_risk.py        족저압 기반 직접 위험 계산
gait_anomaly.py       12가지 이상 패턴 감지
gait_profile.py       개인 기준선 학습 (Welford's algorithm)
foot_zones.py         발 7구역 분석
trend_tracker.py      세션 간 추세 분석
```

---

## 데이터 흐름

### 센서 → 전처리 → 모델 (`src/data/preprocessing.py`)

| 모달리티 | 입력 | 전처리 | 출력 |
|---------|------|--------|------|
| IMU | `(T, 6)` or `(T, 7)` | sanitize → resample → z-score | `(6, 128)` |
| Pressure | `(T, H, W)` or `(T, H*W)` | nan→0 → resample → min-max | `(128, 1, 16, 8)` |
| Skeleton | `(T, J, 3)` or `(T, J, 2)` | sanitize → resample → hip-center | `(3, 128, 17)` |

- `_sanitize()`: Inf → clip to `min(finite_max×2, 1e4)`, NaN → 선형 보간
- 7채널 IMU (타임스탬프 포함) 자동 감지·제거
- 2D 스켈레톤 `(T,J,2)` → z=0 패딩으로 3D 변환

### 실제 데이터 로딩 (`src/data/adapters.py`)

```python
# 폴더 구조
FolderDataAdapter("data/collected/")          # subject_001/imu.csv ...

# NPZ 파일 (imu, pressure, skeleton, labels 키 필수)
NumpyDataAdapter("data/gait.npz")

# CSV glob
CSVDataAdapter(imu_files, pressure_files, skeleton_files, labels)
```

---

## 설정 파일

**`configs/default.yaml`** — 기본 학습·모델 설정:
- `data.sequence_length: 128`
- `data.pressure_grid_size: [16, 8]`
- `data.num_classes: 4`
- `hf_encoders.enabled: false` — `true`로 변경 시 PatchTST + VideoMAE 사용

**`frontend/.env.local`**:
```
NEXT_PUBLIC_API_URL=          # 빈 값 = Next.js 프록시 사용
API_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=
```

**Docker 환경 변수**:
```
LOG_LEVEL=INFO         # DEBUG | INFO | WARNING
API_KEYS=key1,key2     # 빈 값이면 인증 없는 데모 모드
WORKERS=2
```

---

## 재학습

실제 센서 데이터 수집 후 `run_retrain.py` 사용:

```bash
# 권장 (소규모 ≤50 샘플)
python run_retrain.py --data-dir data/collected/ --strategy feature_extraction

# 중규모 (100+ 샘플)
python run_retrain.py --npz data/gait.npz --strategy partial

# CSV glob
python run_retrain.py \
  --imu-files "data/imu_*.csv" \
  --pressure-files "data/pressure_*.csv" \
  --skeleton-files "data/skeleton_*.csv" \
  --labels 0 0 1 1 2 2 3 3
```

전략별 학습 파라미터 비율:
- `feature_extraction`: ~2% (classifier만)
- `partial`: ~96% (마지막 레이어 + head)
- `full`: 100%

---

## 주요 관례

### 단일 소스 원칙

- **보행 클래스 이름** → `src/constants.py::GAIT_CLASS_NAMES` (dict형, api/service.py·run_retrain.py 공유)
- **심각도 라벨** → `src/analysis/common.py::severity_label()` (5단계: 정상/경미/주의/경고/위험)
- **파생 특성 계산** → `src/analysis/common.py::compute_derived_features()` (_features_to_dict에서 위임)
- **수치 안정성 epsilon** → `src/analysis/common.py::EPSILON = 1e-8`

### 모델 입력 배치 형태

```python
{
    "imu":      torch.Tensor,  # (B, 6, 128)
    "pressure": torch.Tensor,  # (B, 128, 1, 16, 8)
    "skeleton": torch.Tensor,  # (B, 3, 128, 17)
}
```

### 병렬 임포트 위치

모든 `import`는 모듈 최상단. 함수 내부 임포트 금지 (train.py·finetune_utils.py 수정 완료).

---

## 알려진 이슈

| 파일 | 내용 |
|------|------|
| `tests/test_disease_biomarkers.py` | `src.analysis.disease_biomarkers` 모듈 없음 — 테스트 실행 시 `--ignore` 필요 |
| `tests/test_api.py` (6건) | disease-risk / analyze 엔드포인트 응답 구조 불일치 (미수정) |

테스트 실행 기본 명령:
```bash
python -m pytest tests/ --ignore=tests/test_disease_biomarkers.py -q
# 정상 기대치: 155 passed, 6 failed
```

---

## 브랜치 규칙

개발 브랜치: `claude/start-mvp-project-UbWB1`

Push:
```bash
git push -u origin claude/start-mvp-project-UbWB1
```
