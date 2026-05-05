"""FastAPI REST API server for multimodal gait analysis.

보행 분석 모델 서빙을 위한 REST API 서버.
모델 로드, 예측, 질병 위험도 평가, 교정 피드백 엔드포인트를 제공합니다.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union, List, Dict

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.preprocessing import preprocess_imu, preprocess_pressure, preprocess_skeleton
from src.models.multimodal_gait_net import MultimodalGaitNet
from src.models.reasoning_engine import GaitReasoningEngine
from src.utils.model_manager import model_manager

logger = logging.getLogger(__name__)

# -- Pydantic request / response models ------------------------------------

GAIT_CLASS_NAMES = ["normal", "antalgic", "ataxic", "parkinsonian"]
GAIT_CLASS_NAMES_KR = ["정상 보행", "절뚝거림", "운동실조", "파킨슨"]


class GaitInput(BaseModel):
    """멀티모달 보행 데이터 입력 스키마."""
    imu: list[list[float]] = Field(..., description="IMU sensor data, shape (T, 6)")
    pressure: list[list[list[float]]] = Field(..., description="Plantar pressure maps, shape (T, H, W)")
    skeleton: list[list[list[float]]] = Field(..., description="Skeleton joint positions, shape (T, J, 3)")
    gait_profile: Optional[str] = Field(None, description="Optional profile for demo/mocking")


class ClassifyResponse(BaseModel):
    prediction: str
    prediction_kr: str
    confidence: float
    class_probabilities: dict[str, float]
    is_demo_mode: bool = False


class DiseaseRisk(BaseModel):
    disease: str
    disease_kr: str
    risk_score: float
    severity: str
    key_signs: list[str]
    referral: str


class DiseaseRiskResponse(BaseModel):
    top_diseases: list[DiseaseRisk]
    ml_prediction: str
    ml_prediction_kr: str
    ml_confidence: float
    ml_top3: list[dict]
    abnormal_biomarkers: list[str]


class InjuryRiskResponse(BaseModel):
    predicted_injury: str
    predicted_injury_kr: str
    confidence: float
    top3: list[dict]
    combined_risk_score: float
    combined_risk_grade: str
    timeline: str
    priority_actions: list[str]


class ReasoningStep(BaseModel):
    step: int
    label: str
    prediction: str
    prediction_kr: str
    probability: float


class ReasoningResponse(BaseModel):
    final_prediction: str
    final_prediction_kr: str
    confidence: float
    reasoning_trace: list[ReasoningStep]
    anomaly_findings: list[dict]
    uncertainty: float
    evidence_strength: float
    report_kr: str
    clinical_notes_kr: Optional[str] = None
    is_demo_mode: bool = False


class FullAnalyzeResponse(BaseModel):
    """프런트엔드 통합 분석 응답."""
    classify: ClassifyResponse
    disease_risk: DiseaseRiskResponse
    injury_risk: InjuryRiskResponse
    reasoning: ReasoningResponse
    model_info: dict


class ModelInfoResponse(BaseModel):
    """모델 정보 응답."""
    id: str
    version: str
    type: str
    metrics: dict
    timestamp: str


class HealthResponse(BaseModel):
    """서버 상태 확인."""
    status: str
    model_loaded: bool
    model_id: Optional[str]
    device: str


# -- Global model state -----------------------------------------------------

_model: Optional[Union[MultimodalGaitNet, GaitReasoningEngine]] = None
_config: Optional[dict] = None
_current_model_id: Optional[str] = None
_device: Optional[torch.device] = None


def _load_config(config_path: str = "configs/default.yaml") -> dict:
    """YAML 설정 파일을 로드합니다."""
    path = Path(config_path)
    if not path.exists():
        # Create a minimal default config if not exists
        return {
            "data": {
                "sequence_length": 128,
                "imu_channels": 6,
                "pressure_grid_size": [16, 8],
                "skeleton_joints": 17,
                "skeleton_dims": 3,
                "num_classes": 4
            },
            "model": {
                "fusion": {"embed_dim": 128},
                "imu_encoder": {"dropout": 0.1},
                "pressure_encoder": {"dropout": 0.1},
                "skeleton_encoder": {"gcn_channels": [64, 128], "temporal_kernel": 9, "dropout": 0.1}
            }
        }
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _init_model(alias_or_id: str = "latest"):
    """모델 매니저를 통해 특정 버전의 모델을 로드합니다."""
    global _model, _config, _device, _current_model_id
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = model_manager.load_checkpoint(alias_or_id, device=str(device))
        _config = checkpoint["config"]
        _current_model_id = checkpoint["model_id"]
        
        m_type = checkpoint.get("model_type", "reasoning")
        if m_type == "basic":
            model = MultimodalGaitNet(_config)
        else:
            model = GaitReasoningEngine(_config)
            
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        _model = model
        _device = device
        logger.info(f"Loaded model: {_current_model_id} on {_device}")
    except Exception as e:
        logger.error(f"Failed to load model {alias_or_id}: {e}")
        if _config is None:
            _config = _load_config()
        _model = GaitReasoningEngine(_config).to(device)
        _device = device
        _current_model_id = "fallback_engine"


# -- App lifecycle -----------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델을 로드합니다."""
    model_id = app.state.model_id if hasattr(app.state, "model_id") else "latest"
    _init_model(model_id)
    yield
    global _model, _config
    _model = None
    _config = None


app = FastAPI(
    title="Shoealls Gait Analysis API",
    description="모델 버전 관리 시스템이 통합된 멀티모달 보행 분석 API",
    version="1.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Payment & Dashboard routers
try:
    from src.serving.payment import router as payment_router
    app.include_router(payment_router)
    # mock_router is now integrated or handled conditionally
except ImportError as e:
    logger.warning(f"Could not load auxiliary routers: {e}")


# -- Endpoints ---------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 및 현재 로드된 모델 상태 확인."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_id=_current_model_id,
        device=str(_device) if _device else "none",
    )


@app.get("/models", response_model=List[ModelInfoResponse])
async def list_models():
    """사용 가능한 모든 모델 목록 조회."""
    return model_manager.list_models()


@app.post("/models/load")
async def load_model(model_id: str = Query(..., description="Model ID or Alias (latest, production)")):
    """특정 버전의 모델을 동적으로 로드합니다."""
    try:
        _init_model(model_id)
        return {"status": "success", "loaded_model": _current_model_id}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/analyze", response_model=FullAnalyzeResponse)
async def analyze_v1(gait_input: GaitInput):
    """프런트엔드 호환 통합 분석 엔드포인트."""
    if not isinstance(_model, GaitReasoningEngine):
        # Fallback to mock if reasoning engine is not available
        try:
            from src.serving.mock_router import analyze as mock_analyze
            return await mock_analyze(gait_input.dict())
        except ImportError:
            raise HTTPException(status_code=503, detail="Reasoning engine unavailable and mock failed")

    # 1. 데이터 전처리
    batch = _prepare_batch(gait_input)

    # 2. 모델 추론
    with torch.no_grad():
        result = _model.reason(batch)
    
    # 3. LLM 리포트 생성
    import asyncio
    report_kr, clinical_notes = await asyncio.gather(
        _model.explain_llm(result),
        _model.explain_clinical(result)
    )

    # 4. 결과 매핑 (프런트엔드 형식)
    i = 0
    pred_idx = result["prediction"][i].item()
    probs = result["calibrated_probs"][i].cpu().numpy()
    
    # Classify mapping
    classify = ClassifyResponse(
        prediction=GAIT_CLASS_NAMES[pred_idx],
        prediction_kr=GAIT_CLASS_NAMES_KR[pred_idx],
        confidence=float(probs[pred_idx]),
        class_probabilities={GAIT_CLASS_NAMES[j]: float(probs[j]) for j in range(len(probs))},
    )

    # Disease Risk mapping (Simplified for now)
    disease_risk = DiseaseRiskResponse(
        top_diseases=[
            DiseaseRisk(
                disease=GAIT_CLASS_NAMES[pred_idx],
                disease_kr=GAIT_CLASS_NAMES_KR[pred_idx],
                risk_score=float(probs[pred_idx]),
                severity="고위험" if probs[pred_idx] > 0.7 else "주의",
                key_signs=[],
                referral="신경과" if pred_idx == 3 else "재활의학과"
            )
        ],
        ml_prediction=GAIT_CLASS_NAMES[pred_idx],
        ml_prediction_kr=GAIT_CLASS_NAMES_KR[pred_idx],
        ml_confidence=float(probs[pred_idx]),
        ml_top3=[{"name_kr": GAIT_CLASS_NAMES_KR[j], "probability": float(probs[j])} for j in np.argsort(probs)[-3:][::-1]],
        abnormal_biomarkers=[]
    )

    # Injury Risk mapping
    injury_risk = InjuryRiskResponse(
        predicted_injury="fall_risk" if pred_idx >= 2 else "none",
        predicted_injury_kr="낙상 위험" if pred_idx >= 2 else "정상",
        confidence=float(result["evidence"]["evidence_strength"][i]),
        top3=[],
        combined_risk_score=float(1.0 - result["uncertainty"][i]),
        combined_risk_grade="고위험" if result["uncertainty"][i] < 0.2 else "보통",
        timeline="즉각적인 주의 필요" if pred_idx >= 2 else "정기 모니터링",
        priority_actions=["균형 운동", "보행 보조기구 검토"] if pred_idx >= 2 else ["규칙적 유산소 운동"]
    )

    # Reasoning mapping
    trace_steps = []
    for s, logits in enumerate(result["diagnosis"]["reasoning_trace"]):
        p = torch.softmax(logits[i], dim=-1)
        top = p.argmax().item()
        trace_steps.append(ReasoningStep(
            step=s,
            label=f"추론 {s}단계" if s > 0 else "초기 가설",
            prediction=GAIT_CLASS_NAMES[top],
            prediction_kr=GAIT_CLASS_NAMES_KR[top],
            probability=float(p[top])
        ))

    anomaly_findings = []
    for m_idx, m_name in enumerate(_model.MODALITY_NAMES_KR):
        scores = result["anomaly_results"][m_idx]["anomaly_scores"][i].cpu().numpy()
        anoms = [{"type": _model.ANOMALY_NAMES_KR[j], "score": float(scores[j])} for j in range(len(scores)) if scores[j] > 0.4]
        if anoms:
            anomaly_findings.append({"modality": m_name, "anomalies": anoms})

    reasoning = ReasoningResponse(
        final_prediction=GAIT_CLASS_NAMES[pred_idx],
        final_prediction_kr=GAIT_CLASS_NAMES_KR[pred_idx],
        confidence=float(probs[pred_idx]),
        reasoning_trace=trace_steps,
        anomaly_findings=anomaly_findings,
        uncertainty=float(result["uncertainty"][i]),
        evidence_strength=float(result["evidence"]["evidence_strength"][i]),
        report_kr=report_kr,
        clinical_notes_kr=clinical_notes,
    )

    return FullAnalyzeResponse(
        classify=classify,
        disease_risk=disease_risk,
        injury_risk=injury_risk,
        reasoning=reasoning,
        model_info={"id": _current_model_id, "device": str(_device)}
    )


def _prepare_batch(gait_input: GaitInput) -> dict[str, torch.Tensor]:
    """입력 데이터를 전처리하여 모델 입력 배치로 변환."""
    data_cfg = _config["data"]
    seq_len = data_cfg["sequence_length"]
    grid_size = tuple(data_cfg["pressure_grid_size"])
    num_joints = data_cfg["skeleton_joints"]

    imu_processed = preprocess_imu(np.array(gait_input.imu, dtype=np.float32), target_length=seq_len)
    pressure_processed = preprocess_pressure(np.array(gait_input.pressure, dtype=np.float32), target_length=seq_len, grid_size=grid_size)
    skeleton_processed = preprocess_skeleton(np.array(gait_input.skeleton, dtype=np.float32), target_length=seq_len, num_joints=num_joints)

    return {
        "imu": torch.from_numpy(imu_processed).unsqueeze(0).to(_device),
        "pressure": torch.from_numpy(pressure_processed).unsqueeze(0).to(_device),
        "skeleton": torch.from_numpy(skeleton_processed).unsqueeze(0).to(_device),
    }

# -- CLI entrypoint ----------------------------------------------------------

def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Shoealls Gait Analysis API Server")
    parser.add_argument("--model-id", type=str, default="latest", help="Initial model ID or alias to load")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    app.state.model_id = args.model_id

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
