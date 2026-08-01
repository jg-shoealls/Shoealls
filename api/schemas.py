"""API request/response schemas."""

from typing import Optional
from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────

class SensorData(BaseModel):
    """Raw multimodal sensor input."""
    imu: list[list[float]] = Field(
        description="IMU data [seq_len, 6] — accel(x,y,z) + gyro(x,y,z)"
    )
    pressure: list[list[float]] = Field(
        description="Foot pressure grid [H, W], default 16×8"
    )
    skeleton: list[list[list[float]]] = Field(
        description="Skeleton joint data [seq_len, 17, 3] — (x,y,z) per joint"
    )


class GaitFeatures(BaseModel):
    """Pre-extracted gait biomarker features for analysis endpoints."""
    gait_speed: float = Field(description="보행 속도 (m/s), 정상 1.0~1.4")
    cadence: float = Field(description="분당 걸음수 (steps/min), 정상 100~130")
    stride_regularity: float = Field(description="보폭 규칙성 (0~1), 정상 0.7+")
    step_symmetry: float = Field(description="좌우 대칭성 (0~1), 정상 0.85+")
    cop_sway: float = Field(description="체중심 흔들림 (normalized), 정상 <0.06")
    ml_index: float = Field(description="ML (좌우) 압력 지수 (0~1)")
    arch_index: float = Field(description="아치 지수 (0~1)")
    acceleration_rms: float = Field(description="가속도 RMS")
    zone_heel_medial_mean: float = Field(default=0.30, description="뒤꿈치 내측 압력")
    zone_heel_lateral_mean: float = Field(default=0.28, description="뒤꿈치 외측 압력")
    zone_forefoot_medial_mean: float = Field(default=0.33, description="전족부 내측 압력")
    zone_forefoot_lateral_mean: float = Field(default=0.30, description="전족부 외측 압력")
    zone_toes_mean: float = Field(default=0.18, description="발가락 압력")
    zone_midfoot_medial_mean: float = Field(default=0.10, description="중족부 내측 압력")
    zone_midfoot_lateral_mean: float = Field(default=0.08, description="중족부 외측 압력")
    # Optional extras for injury risk
    ml_variability: Optional[float] = Field(default=None)
    heel_pressure_ratio: Optional[float] = Field(default=None)
    forefoot_pressure_ratio: Optional[float] = Field(default=None)
    pressure_asymmetry: Optional[float] = Field(default=None)
    trunk_sway: Optional[float] = Field(default=None)


class ClassifyRequest(BaseModel):
    sensor_data: SensorData
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="사전학습 모델 경로 (없으면 무작위 초기화 데모 모드)"
    )


class DiseaseRiskRequest(BaseModel):
    features: GaitFeatures


class InjuryRiskRequest(BaseModel):
    features: GaitFeatures


class ReasoningRequest(BaseModel):
    sensor_data: SensorData
    checkpoint_path: Optional[str] = Field(default=None)


class AnalyzeRequest(BaseModel):
    """종합 분석: 센서 데이터 + 보행 특성 모두 필요."""
    sensor_data: SensorData
    features: GaitFeatures
    checkpoint_path: Optional[str] = Field(default=None)


# ── Response Models ───────────────────────────────────────────────────

class GaitClassifyResponse(BaseModel):
    prediction: str
    prediction_kr: str
    confidence: float
    class_probabilities: dict[str, float]
    is_demo_mode: bool = Field(description="체크포인트 없는 데모 모드 여부")


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
    body_risk_map: dict[str, float]
    timeline: str
    combined_risk_score: float
    combined_risk_grade: str
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
    uncertainty: float
    anomaly_findings: list[dict]
    modality_weights: dict[str, float]
    evidence_strength: float
    reasoning_trace: list[ReasoningStep]
    report_kr: str
    is_demo_mode: bool


class AnalyzeResponse(BaseModel):
    classify: GaitClassifyResponse
    disease_risk: DiseaseRiskResponse
    injury_risk: InjuryRiskResponse
    reasoning: ReasoningResponse
