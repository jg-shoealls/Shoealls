"""Pydantic request/response schemas for the inference API."""

from pydantic import BaseModel, Field


# ── Request Schemas ──────────────────────────────────────────────────

class SensorData(BaseModel):
    """Raw sensor data for a single gait session."""
    imu: list[list[float]] = Field(
        ...,
        description="IMU data as (6, T) — 6 channels (accel xyz + gyro xyz) x T timesteps",
    )
    pressure: list[list[list[list[float]]]] = Field(
        ...,
        description="Pressure data as (T, 1, 16, 8) — T frames of 16x8 plantar pressure grid",
    )


class GaitFeatures(BaseModel):
    """Pre-extracted gait feature vector (alternative to raw sensor data)."""
    gait_speed: float = Field(..., ge=0, description="m/s")
    cadence: float = Field(..., ge=0, description="steps/min")
    stride_regularity: float = Field(..., ge=0, le=1)
    step_symmetry: float = Field(..., ge=0, le=1)
    cop_sway: float = Field(..., ge=0)
    ml_index: float = Field(...)
    arch_index: float = Field(..., ge=0, le=1)
    acceleration_rms: float = Field(..., ge=0)
    zone_heel_medial_mean: float = Field(..., ge=0)
    zone_heel_lateral_mean: float = Field(..., ge=0)
    zone_forefoot_medial_mean: float = Field(..., ge=0)
    zone_forefoot_lateral_mean: float = Field(..., ge=0)
    zone_toes_mean: float = Field(..., ge=0)
    zone_midfoot_medial_mean: float = Field(..., ge=0)
    zone_midfoot_lateral_mean: float = Field(..., ge=0)


class PredictRequest(BaseModel):
    """Request for /predict endpoint — accepts raw sensor data or pre-extracted features."""
    sensor_data: SensorData | None = None
    features: GaitFeatures | None = None


class DiseaseRiskRequest(BaseModel):
    """Request for /disease-risk endpoint."""
    sensor_data: SensorData | None = None
    features: GaitFeatures | None = None


class FeedbackRequest(BaseModel):
    """Request for /feedback endpoint."""
    sensor_data: SensorData | None = None
    features: GaitFeatures | None = None


# ── Response Schemas ─────────────────────────────────────────────────

class AnomalyPatternResponse(BaseModel):
    pattern_id: str
    korean_name: str
    severity: float
    severity_label: str
    description: str
    correction: str


class PredictResponse(BaseModel):
    """Response from /predict endpoint — gait classification + anomaly detection."""
    anomaly_score: float = Field(..., description="Overall anomaly score 0-1")
    anomaly_grade: str = Field(..., description="Grade: normal/mild/caution/warning/danger")
    abnormal_patterns: list[AnomalyPatternResponse]
    gait_features: dict[str, float] = Field(..., description="Extracted gait feature values")


class DiseaseRiskItemResponse(BaseModel):
    disease_id: str
    korean_name: str
    risk_score: float
    severity: str
    confidence: float
    matched_signs: list[str]
    referral: str


class BiomarkerResponse(BaseModel):
    name: str
    korean_name: str
    value: float
    normal_range: list[float]
    unit: str
    is_abnormal: bool


class DiseaseRiskResponse(BaseModel):
    """Response from /disease-risk endpoint."""
    overall_health_score: float
    top_risks: list[DiseaseRiskItemResponse]
    all_results: list[DiseaseRiskItemResponse]
    biomarkers: list[BiomarkerResponse]


class FeedbackItemResponse(BaseModel):
    category: str
    priority: int
    title: str
    description: str
    exercises: list[str]


class InjuryRiskItemResponse(BaseModel):
    name: str
    korean_name: str
    risk_score: float
    severity: str
    contributing_factors: list[str]
    recommendation: str


class FeedbackResponse(BaseModel):
    """Response from /feedback endpoint."""
    overall_status: str
    encouragement: str
    feedback_items: list[FeedbackItemResponse]
    injury_risks: list[InjuryRiskItemResponse]
