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


class GaitInput(BaseModel):
    """멀티모달 보행 데이터 입력 스키마."""
    imu: list[list[float]] = Field(..., description="IMU sensor data, shape (T, 6)")
    pressure: list[list[list[float]]] = Field(..., description="Plantar pressure maps, shape (T, H, W)")
    skeleton: list[list[list[float]]] = Field(..., description="Skeleton joint positions, shape (T, J, 3)")


class PredictionResponse(BaseModel):
    """보행 분류 예측 결과."""
    predicted_class: str
    class_index: int
    probabilities: dict[str, float]


class AnalyzeResponse(BaseModel):
    """정밀 추론 및 LLM 리포트 결과."""
    prediction: str
    confidence: float
    uncertainty: float
    report_kr: str
    clinical_notes_kr: Optional[str] = None
    reasoning_steps: list[str]
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
        raise FileNotFoundError(f"Config not found: {config_path}")
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
        
        # 모델 타입 결정
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
        # Fallback to empty model if requested by app state or config
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
    version="1.2.0",
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
    from src.serving.mock_router import router as mock_router
    app.include_router(mock_router)
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


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(gait_input: GaitInput):
    """정밀 추론 엔진을 사용하여 분석하고 리포트를 생성합니다."""
    if not isinstance(_model, GaitReasoningEngine):
        raise HTTPException(status_code=400, detail="Current model does not support reasoning")

    batch = _prepare_batch(gait_input)

    with torch.no_grad():
        result = _model.reason(batch)
    
    import asyncio
    report_kr, clinical_notes = await asyncio.gather(
        _model.explain_llm(result),
        _model.explain_clinical(result)
    )

    pred_idx = result["prediction"][0].item()
    trace_steps = []
    for s, logits in enumerate(result["diagnosis"]["reasoning_trace"]):
        p = torch.softmax(logits[0], dim=-1)
        top = p.argmax().item()
        trace_steps.append(f"Step {s}: {_model.CLASS_NAMES_KR[top]} ({p[top]:.1%})")

    return AnalyzeResponse(
        prediction=_model.CLASS_NAMES_KR[pred_idx],
        confidence=float(result["calibrated_probs"][0][pred_idx]),
        uncertainty=float(result["uncertainty"][0]),
        report_kr=report_kr,
        clinical_notes_kr=clinical_notes,
        reasoning_steps=trace_steps,
        model_info={
            "id": _current_model_id,
            "config": _config["model"]
        }
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
