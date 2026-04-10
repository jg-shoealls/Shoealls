"""Shoealls MVP — FastAPI REST API server.

보행 데이터 분석 AI 서비스:
  - 보행 패턴 분류 (MultimodalGaitNet)
  - 질환 위험도 예측 (규칙기반 + ML 앙상블)
  - 부상 위험 예측 (ML + 패턴감지)
  - Chain-of-Reasoning 추론 분석 (GaitReasoningEngine)

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .schemas import (
    ClassifyRequest,
    DiseaseRiskRequest,
    InjuryRiskRequest,
    ReasoningRequest,
    AnalyzeRequest,
    GaitClassifyResponse,
    DiseaseRiskResponse,
    InjuryRiskResponse,
    ReasoningResponse,
    AnalyzeResponse,
)
from .service import get_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up sklearn models on synthetic data
    svc = get_service()
    svc.warmup()
    yield


app = FastAPI(
    title="Shoealls Gait Analysis API",
    description=(
        "멀티모달 보행 AI 분석 REST API\n\n"
        "**센서 입력**: IMU (6ch) + 족저압 (16×8) + 스켈레톤 (17 joints × 3D)\n\n"
        "**분석 기능**: 보행 패턴 분류 / 질환 위험도 / 부상 위험 / Chain-of-Reasoning"
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Health ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """서버 상태 확인."""
    return {"status": "ok", "service": "shoealls-gait-api", "version": "0.1.0"}


# ── Gait Classification ────────────────────────────────────────────────

@app.post(
    "/api/v1/classify",
    response_model=GaitClassifyResponse,
    tags=["Analysis"],
    summary="보행 패턴 분류",
    description=(
        "IMU + 족저압 + 스켈레톤 센서 데이터로 보행 패턴을 분류합니다.\n\n"
        "**분류 클래스**: 정상 보행 / 절뚝거림 / 운동실조 / 파킨슨\n\n"
        "> 체크포인트 없으면 랜덤 초기화 데모 모드로 실행됩니다."
    ),
)
def classify_gait(req: ClassifyRequest):
    try:
        svc = get_service()
        return svc.classify(req.sensor_data, req.checkpoint_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Disease Risk ───────────────────────────────────────────────────────

@app.post(
    "/api/v1/disease-risk",
    response_model=DiseaseRiskResponse,
    tags=["Analysis"],
    summary="질환 위험도 예측",
    description=(
        "보행 바이오마커로 14개 질환 위험도를 평가합니다.\n\n"
        "**질환 카테고리**: 신경계(파킨슨, 치매, 다발성경화증, 소뇌실조증) | "
        "뇌혈관계(뇌졸중, 뇌출혈, 뇌경색) | "
        "근골격계(골관절염, 류마티스, 디스크, 척추관협착증) | "
        "기타(당뇨신경병증, 말초동맥질환, 전정기관장애)\n\n"
        "규칙 기반 위험도 + ML 앙상블 분류 결과를 함께 제공합니다."
    ),
)
def predict_disease_risk(req: DiseaseRiskRequest):
    try:
        svc = get_service()
        return svc.disease_risk(req.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Injury Risk ────────────────────────────────────────────────────────

@app.post(
    "/api/v1/injury-risk",
    response_model=InjuryRiskResponse,
    tags=["Analysis"],
    summary="부상 위험 예측",
    description=(
        "보행 패턴 기반 ML 부상 위험 예측.\n\n"
        "**예측 부상**: 족저근막염 / 중족골 피로골절 / 발목 염좌 / "
        "무릎 과부하 / 고관절·요통 / 낙상 위험 / 아킬레스건염 / 경골 스트레스\n\n"
        "이상 패턴 감지(40%) + ML 예측(60%) 종합 위험 점수를 제공합니다."
    ),
)
def predict_injury_risk(req: InjuryRiskRequest):
    try:
        svc = get_service()
        return svc.injury_risk(req.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Chain-of-Reasoning ────────────────────────────────────────────────

@app.post(
    "/api/v1/reasoning",
    response_model=ReasoningResponse,
    tags=["Analysis"],
    summary="Chain-of-Reasoning 추론 분석",
    description=(
        "4단계 추론 체인으로 보행을 심층 분석합니다.\n\n"
        "1. **이상 감지** — 모달리티별 비정상 패턴 탐지\n"
        "2. **근거 수집** — 교차 모달 검증으로 근거 강화\n"
        "3. **감별 진단** — 가설 생성→대조→업데이트 반복\n"
        "4. **신뢰도 보정** — 추론 일관성 기반 최종 판정\n\n"
        "> 체크포인트 없으면 랜덤 초기화 데모 모드로 실행됩니다."
    ),
)
def reasoning_analysis(req: ReasoningRequest):
    try:
        svc = get_service()
        return svc.reasoning(req.sensor_data, req.checkpoint_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Full Analysis ─────────────────────────────────────────────────────

@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="종합 분석 (모든 기능)",
    description=(
        "4가지 분석을 한 번에 실행합니다:\n"
        "보행 분류 + 질환 위험 + 부상 위험 + Chain-of-Reasoning"
    ),
)
def full_analysis(req: AnalyzeRequest):
    try:
        svc = get_service()
        return svc.analyze(req.sensor_data, req.features, req.checkpoint_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
