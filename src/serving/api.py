"""FastAPI REST API server for multimodal gait analysis.

보행 분석 모델 서빙을 위한 REST API 서버.
모델 로드, 예측, 질병 위험도 평가, 교정 피드백 엔드포인트를 제공합니다.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.preprocessing import preprocess_imu, preprocess_pressure, preprocess_skeleton
from src.models.multimodal_gait_net import MultimodalGaitNet

logger = logging.getLogger(__name__)

# -- Pydantic request / response models ------------------------------------

GAIT_CLASS_NAMES = ["normal", "antalgic", "ataxic", "parkinsonian"]


class GaitInput(BaseModel):
    """멀티모달 보행 데이터 입력 스키마.

    각 센서 데이터는 중첩 리스트(nested list)로 전달됩니다.
    """

    imu: list[list[float]] = Field(
        ...,
        description="IMU sensor data, shape (T, 6): [ax, ay, az, gx, gy, gz] per timestep",
    )
    pressure: list[list[list[float]]] = Field(
        ...,
        description="Plantar pressure maps, shape (T, H, W)",
    )
    skeleton: list[list[list[float]]] = Field(
        ...,
        description="Skeleton joint positions, shape (T, J, 3): [x, y, z] per joint",
    )


class PredictionResponse(BaseModel):
    """보행 분류 예측 결과."""

    predicted_class: str
    class_index: int
    probabilities: dict[str, float]


class DiseaseRiskResponse(BaseModel):
    """질병 위험도 예측 결과."""

    disease_probabilities: dict[str, float]
    severity_score: float
    risk_level: str


class FeedbackResponse(BaseModel):
    """교정 피드백 결과."""

    predicted_class: str
    risk_level: str
    recommendations: list[str]


class HealthResponse(BaseModel):
    """서버 상태 확인."""

    status: str
    model_loaded: bool
    device: str


# -- Global model state -----------------------------------------------------

_model: Optional[MultimodalGaitNet] = None
_config: Optional[dict] = None
_device: Optional[torch.device] = None


def _load_config(config_path: str = "configs/default.yaml") -> dict:
    """YAML 설정 파일을 로드합니다."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_model(config: dict, checkpoint_path: Optional[str] = None) -> MultimodalGaitNet:
    """모델을 생성하고, 체크포인트가 있으면 가중치를 로드합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalGaitNet(config)

    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint from %s", checkpoint_path)

    model.to(device)
    model.eval()
    return model


def _prepare_batch(gait_input: GaitInput, config: dict) -> dict[str, torch.Tensor]:
    """입력 데이터를 전처리하여 모델 입력 배치로 변환합니다.

    Preprocess raw sensor arrays into a model-ready batch dictionary.
    """
    data_cfg = config["data"]
    seq_len = data_cfg["sequence_length"]
    grid_size = tuple(data_cfg["pressure_grid_size"])
    num_joints = data_cfg["skeleton_joints"]

    imu_arr = np.array(gait_input.imu, dtype=np.float32)
    pressure_arr = np.array(gait_input.pressure, dtype=np.float32)
    skeleton_arr = np.array(gait_input.skeleton, dtype=np.float32)

    imu_processed = preprocess_imu(imu_arr, target_length=seq_len)
    pressure_processed = preprocess_pressure(pressure_arr, target_length=seq_len, grid_size=grid_size)
    skeleton_processed = preprocess_skeleton(skeleton_arr, target_length=seq_len, num_joints=num_joints)

    device = next(_model.parameters()).device
    return {
        "imu": torch.from_numpy(imu_processed).unsqueeze(0).to(device),
        "pressure": torch.from_numpy(pressure_processed).unsqueeze(0).to(device),
        "skeleton": torch.from_numpy(skeleton_processed).unsqueeze(0).to(device),
    }


# -- App lifecycle -----------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델을 로드하고, 종료 시 정리합니다."""
    global _model, _config, _device

    config_path = app.state.config_path if hasattr(app.state, "config_path") else "configs/default.yaml"
    checkpoint_path = app.state.checkpoint_path if hasattr(app.state, "checkpoint_path") else None

    _config = _load_config(config_path)
    _model = _load_model(_config, checkpoint_path)
    _device = next(_model.parameters()).device

    logger.info("Model loaded on %s (%d params)", _device, _model.get_num_params())
    yield

    _model = None
    _config = None
    logger.info("Model unloaded")


app = FastAPI(
    title="Multimodal Gait Analysis API",
    description="멀티모달 보행 분석 AI 모델 서빙 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Payment API router
from src.serving.payment import router as payment_router
app.include_router(payment_router)

# Dashboard API mock router
from src.serving.mock_router import router as mock_router
app.include_router(mock_router)


# -- Endpoints ---------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 및 모델 상태를 확인합니다. / Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        device=str(_device) if _device else "none",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(gait_input: GaitInput):
    """보행 패턴을 분류합니다. / Predict gait class from multimodal sensor data."""
    if _model is None or _config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        batch = _prepare_batch(gait_input, _config)
    except (ValueError, IndexError) as e:
        raise HTTPException(status_code=422, detail=f"Input preprocessing failed: {e}")

    with torch.no_grad():
        logits = _model(batch)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    class_idx = int(probs.argmax())
    num_classes = _config["data"]["num_classes"]
    class_names = GAIT_CLASS_NAMES[:num_classes]

    prob_dict = {name: round(float(probs[i]), 4) for i, name in enumerate(class_names)}

    return PredictionResponse(
        predicted_class=class_names[class_idx],
        class_index=class_idx,
        probabilities=prob_dict,
    )


@app.post("/disease-risk", response_model=DiseaseRiskResponse)
async def disease_risk(gait_input: GaitInput):
    """질병 위험도를 예측합니다. / Predict disease risk probabilities from gait data."""
    if _model is None or _config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        batch = _prepare_batch(gait_input, _config)
    except (ValueError, IndexError) as e:
        raise HTTPException(status_code=422, detail=f"Input preprocessing failed: {e}")

    with torch.no_grad():
        logits = _model(batch)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    class_idx = int(probs.argmax())
    max_prob = float(probs.max())

    disease_names = ["normal", "alzheimer", "parkinson", "dementia"]
    num_classes = min(len(disease_names), probs.shape[0])
    disease_probs = {disease_names[i]: round(float(probs[i]), 4) for i in range(num_classes)}

    if max_prob >= 0.75:
        risk_level = "high"
    elif max_prob >= 0.5:
        risk_level = "moderate"
    elif max_prob >= 0.25:
        risk_level = "low"
    else:
        risk_level = "minimal"

    severity = round(1.0 - float(probs[0]) if num_classes > 0 else 0.0, 4)

    return DiseaseRiskResponse(
        disease_probabilities=disease_probs,
        severity_score=severity,
        risk_level=risk_level,
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(gait_input: GaitInput):
    """교정 피드백을 생성합니다. / Generate corrective feedback from gait analysis."""
    if _model is None or _config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        batch = _prepare_batch(gait_input, _config)
    except (ValueError, IndexError) as e:
        raise HTTPException(status_code=422, detail=f"Input preprocessing failed: {e}")

    with torch.no_grad():
        logits = _model(batch)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    class_idx = int(probs.argmax())
    num_classes = _config["data"]["num_classes"]
    class_names = GAIT_CLASS_NAMES[:num_classes]
    predicted_class = class_names[class_idx]

    recommendations = _generate_recommendations(predicted_class, probs)
    max_abnormal = max(float(probs[i]) for i in range(1, num_classes)) if num_classes > 1 else 0.0

    if max_abnormal >= 0.6:
        risk_level = "high"
    elif max_abnormal >= 0.3:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return FeedbackResponse(
        predicted_class=predicted_class,
        risk_level=risk_level,
        recommendations=recommendations,
    )


def _generate_recommendations(predicted_class: str, probs: torch.Tensor) -> list[str]:
    """분류 결과에 기반한 교정 권장 사항을 생성합니다.

    Generate corrective recommendations based on the predicted gait class.
    """
    recommendations_map = {
        "normal": [
            "현재 정상적인 보행 패턴입니다. 규칙적인 운동을 유지하세요.",
            "주기적인 보행 모니터링을 권장합니다.",
        ],
        "antalgic": [
            "통증 회피 보행 패턴이 감지되었습니다.",
            "체중 분산 운동: 양발에 균등한 체중 부하 연습 (10회 3세트)",
            "하지 근력 강화: 의자 스쿼트 (10회 3세트)",
            "정형외과 전문의 상담을 권장합니다.",
        ],
        "ataxic": [
            "균형 불안정 보행 패턴이 감지되었습니다.",
            "균형 훈련: 한 발 서기 (30초씩 좌우 번갈아, 5세트)",
            "일직선 걷기 훈련 (10걸음씩 3회)",
            "신경과 전문의 상담을 권장합니다.",
        ],
        "parkinsonian": [
            "파킨슨성 보행 패턴이 감지되었습니다.",
            "보폭 훈련: 바닥 표시선에 맞춰 큰 보폭으로 걷기 (5분)",
            "리듬 보행: 메트로놈에 맞춰 걷기 연습 (BPM 100-110)",
            "팔 흔들기 의식적 연습: 걸을 때 양팔 크게 흔들기",
            "신경과 전문의 상담을 강력히 권장합니다.",
        ],
    }

    return recommendations_map.get(predicted_class, [
        "보행 분석 결과에 따른 전문의 상담을 권장합니다.",
    ])


# -- CLI entrypoint ----------------------------------------------------------

def main():
    """Uvicorn 기반 서버 실행. / Run the API server with uvicorn."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Gait Analysis API Server")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    app.state.config_path = args.config
    app.state.checkpoint_path = args.checkpoint

    uvicorn.run(
        "src.serving.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
