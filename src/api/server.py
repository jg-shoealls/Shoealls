"""FastAPI inference server for real-time gait analysis.

Endpoints:
    POST /predict      — Sensor data -> gait anomaly classification
    POST /disease-risk — Gait features -> disease risk screening (14 diseases)
    POST /feedback     — Gait features -> personalized corrective feedback
    POST /parkinsons   — Parkinson's disease-focused gait analysis
    GET  /health       — Server health check
"""

import numpy as np
from fastapi import FastAPI, HTTPException

from src.analysis.gait_profile import PersonalGaitProfiler
from src.analysis.gait_anomaly import GaitAnomalyDetector
from src.analysis.disease_predictor import DiseaseRiskPredictor
from src.analysis.injury_risk import InjuryRiskEngine
from src.analysis.feedback import CorrektiveFeedbackGenerator
from src.analysis.parkinsons_analyzer import ParkinsonsAnalyzer

from .schemas import (
    PredictRequest,
    PredictResponse,
    AnomalyPatternResponse,
    DiseaseRiskRequest,
    DiseaseRiskResponse,
    DiseaseRiskItemResponse,
    BiomarkerResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackItemResponse,
    InjuryRiskItemResponse,
    ParkinsonsRequest,
    ParkinsonsResponse,
    SubPatternResponse,
    SubPatternIndicatorResponse,
)

app = FastAPI(
    title="Shoealls Gait Analysis API",
    description="실시간 보행 분석 추론 API — 보행 분류, 질병 위험도, 교정 피드백",
    version="1.0.0",
)

# ── Shared analysis engine singletons ────────────────────────────────
profiler = PersonalGaitProfiler(grid_h=16, grid_w=8)
anomaly_detector = GaitAnomalyDetector()
disease_predictor = DiseaseRiskPredictor()
injury_engine = InjuryRiskEngine(grid_h=16, grid_w=8)
feedback_generator = CorrektiveFeedbackGenerator()
parkinsons_analyzer = ParkinsonsAnalyzer()


# ── Helpers ──────────────────────────────────────────────────────────

def _extract_features(request) -> tuple[dict[str, float], np.ndarray | None]:
    """Extract gait features from request, returning (features_dict, pressure_array_or_None).

    Accepts either raw sensor_data or pre-extracted features.
    """
    if request.features is not None:
        return request.features.model_dump(), None

    if request.sensor_data is not None:
        try:
            pressure = np.array(request.sensor_data.pressure, dtype=np.float32)
            imu = np.array(request.sensor_data.imu, dtype=np.float32)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid sensor data format: {e}")

        if pressure.ndim == 3:
            pressure = pressure[:, np.newaxis, :, :]
        if pressure.ndim != 4 or pressure.shape[1] != 1:
            raise HTTPException(
                status_code=422,
                detail=f"Pressure must be (T, 1, 16, 8), got {pressure.shape}",
            )
        if imu.ndim != 2 or imu.shape[0] != 6:
            raise HTTPException(
                status_code=422,
                detail=f"IMU must be (6, T), got {imu.shape}",
            )

        features = profiler.extract_session_features(pressure, imu)
        return features, pressure

    raise HTTPException(
        status_code=422,
        detail="Either 'sensor_data' or 'features' must be provided.",
    )


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Server health check."""
    return {"status": "ok", "service": "shoealls-gait-api"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Classify gait patterns and detect anomalies from sensor data.

    Accepts raw IMU + pressure sensor data or pre-extracted gait features.
    Returns anomaly score, grade, detected abnormal patterns, and extracted features.
    """
    features, _ = _extract_features(request)

    report = anomaly_detector.detect(features)

    abnormal = [
        AnomalyPatternResponse(
            pattern_id=p.pattern_id,
            korean_name=p.korean_name,
            severity=round(p.severity, 4),
            severity_label=p.severity_label,
            description=p.description,
            correction=p.correction,
        )
        for p in report.abnormal_patterns
    ]

    return PredictResponse(
        anomaly_score=round(report.anomaly_score, 4),
        anomaly_grade=report.anomaly_grade,
        abnormal_patterns=abnormal,
        gait_features={k: round(v, 6) for k, v in features.items()},
    )


@app.post("/disease-risk", response_model=DiseaseRiskResponse)
def disease_risk(request: DiseaseRiskRequest):
    """Screen for 14 disease risks based on gait biomarkers.

    Returns per-disease risk scores, biomarker profile, and overall health score.
    """
    features, _ = _extract_features(request)

    report = disease_predictor.predict(features)

    top_risks = [
        DiseaseRiskItemResponse(
            disease_id=r.disease_id,
            korean_name=r.korean_name,
            risk_score=round(r.risk_score, 4),
            severity=r.severity,
            confidence=round(r.confidence, 4),
            matched_signs=r.matched_signs,
            referral=r.referral,
        )
        for r in report.top_risks
    ]

    all_results = [
        DiseaseRiskItemResponse(
            disease_id=r.disease_id,
            korean_name=r.korean_name,
            risk_score=round(r.risk_score, 4),
            severity=r.severity,
            confidence=round(r.confidence, 4),
            matched_signs=r.matched_signs,
            referral=r.referral,
        )
        for r in report.results
    ]

    biomarkers = [
        BiomarkerResponse(
            name=b.name,
            korean_name=b.korean_name,
            value=round(b.value, 6),
            normal_range=list(b.normal_range),
            unit=b.unit,
            is_abnormal=b.is_abnormal,
        )
        for b in report.biomarker_profile.biomarkers
    ]

    return DiseaseRiskResponse(
        overall_health_score=report.overall_health_score,
        top_risks=top_risks,
        all_results=all_results,
        biomarkers=biomarkers,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    """Generate personalized corrective feedback based on gait analysis.

    Combines injury risk assessment with gait deviation analysis to produce
    prioritized exercise, footwear, posture, and medical recommendations.
    """
    features, pressure = _extract_features(request)

    # Injury assessment needs raw pressure data; generate synthetic if only features provided
    if pressure is not None:
        injury_report = injury_engine.assess_risk(pressure)
    else:
        injury_report = _injury_from_features(features)

    fb = feedback_generator.generate(injury_report)

    feedback_items = [
        FeedbackItemResponse(
            category=item.category,
            priority=item.priority,
            title=item.title,
            description=item.description,
            exercises=item.exercises,
        )
        for item in fb.items
    ]

    injury_risks = [
        InjuryRiskItemResponse(
            name=r.name,
            korean_name=r.korean_name,
            risk_score=round(r.risk_score, 4),
            severity=r.severity,
            contributing_factors=r.contributing_factors,
            recommendation=r.recommendation,
        )
        for r in injury_report.risks
    ]

    return FeedbackResponse(
        overall_status=fb.overall_status,
        encouragement=fb.encouragement,
        feedback_items=feedback_items,
        injury_risks=injury_risks,
    )


@app.post("/parkinsons", response_model=ParkinsonsResponse)
def parkinsons(request: ParkinsonsRequest):
    """Parkinson's disease-focused gait analysis.

    Detects 5 Parkinson's-specific gait sub-patterns (shuffling, freezing,
    festination, postural instability, bradykinesia), estimates Hoehn & Yahr
    stage, and provides targeted clinical recommendations.
    """
    features, _ = _extract_features(request)

    report = parkinsons_analyzer.analyze(features)

    def _convert_sub_pattern(r):
        return SubPatternResponse(
            pattern_id=r.pattern_id,
            korean_name=r.korean_name,
            score=round(r.score, 4),
            detected=r.detected,
            description=r.description,
            clinical_meaning=r.clinical_meaning,
            indicator_details=[
                SubPatternIndicatorResponse(**d) for d in r.indicator_details
            ],
        )

    return ParkinsonsResponse(
        risk_score=report.risk_score,
        risk_label=report.risk_label,
        hoehn_yahr_stage=report.hoehn_yahr_stage,
        hoehn_yahr_label=report.hoehn_yahr_label,
        hoehn_yahr_description=report.hoehn_yahr_description,
        sub_patterns=[_convert_sub_pattern(r) for r in report.sub_patterns],
        detected_patterns=[_convert_sub_pattern(r) for r in report.detected_patterns],
        key_findings=report.key_findings,
        recommendations=report.recommendations,
        confidence=report.confidence,
    )


def _injury_from_features(features: dict[str, float]):
    """Synthesize a minimal pressure sequence from extracted features for injury assessment."""
    T, H, W = 32, 16, 8
    pressure = np.ones((T, 1, H, W), dtype=np.float32) * 0.1

    heel_mean = (features.get("zone_heel_medial_mean", 0.3) + features.get("zone_heel_lateral_mean", 0.3)) / 2
    ff_mean = (features.get("zone_forefoot_medial_mean", 0.3) + features.get("zone_forefoot_lateral_mean", 0.3)) / 2
    toe_mean = features.get("zone_toes_mean", 0.15)
    mid_mean = (features.get("zone_midfoot_medial_mean", 0.1) + features.get("zone_midfoot_lateral_mean", 0.08)) / 2

    pressure[:, 0, 11:16, :] = heel_mean
    pressure[:, 0, 3:7, :] = ff_mean
    pressure[:, 0, 0:3, :] = toe_mean
    pressure[:, 0, 7:11, :] = mid_mean

    ml = features.get("ml_index", 0.0)
    if ml > 0:
        pressure[:, 0, :, 4:8] *= (1 + ml)
    else:
        pressure[:, 0, :, 0:4] *= (1 - ml)

    return injury_engine.assess_risk(pressure)
