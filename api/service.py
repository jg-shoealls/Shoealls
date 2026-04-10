"""ML service layer: wraps all analysis modules for API use."""

from __future__ import annotations

import numpy as np
import torch
import yaml
from pathlib import Path

from src.models.multimodal_gait_net import MultimodalGaitNet
from src.models.reasoning_engine import GaitReasoningEngine
from src.analysis.disease_predictor import DiseaseRiskPredictor
from src.analysis.disease_classifier import GaitDiseaseClassifier
from src.analysis.injury_predictor import InjuryRiskPredictor
from src.data.preprocessing import preprocess_imu, preprocess_pressure, preprocess_skeleton

from .schemas import (
    SensorData, GaitFeatures,
    GaitClassifyResponse, DiseaseRiskResponse, DiseaseRisk,
    InjuryRiskResponse, ReasoningResponse, ReasoningStep, AnalyzeResponse,
)

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

GAIT_CLASS_NAMES = {
    0: ("normal", "정상 보행"),
    1: ("antalgic", "절뚝거림"),
    2: ("ataxic", "운동실조"),
    3: ("parkinsonian", "파킨슨"),
}

MODALITY_NAMES = ["IMU (관성센서)", "족저압 센서", "스켈레톤"]

ANOMALY_NAMES = [
    "좌우 비대칭", "리듬 불규칙", "진폭 이상", "주파수 이상",
    "공간 패턴 이상", "시간 지연", "떨림", "보행 동결",
]


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _sensor_to_tensors(data: SensorData, config: dict) -> dict:
    """Convert SensorData → model-ready torch tensors (batch size 1).

    Uses standard preprocessing pipeline:
      imu      (T,6)    → preprocess_imu      → (6,T)       → (1,6,T)
      pressure (T,H,W)  → preprocess_pressure → (T,1,H,W)   → (1,T,1,H,W)
      skeleton (T,J,3)  → preprocess_skeleton → (3,T,J)     → (1,3,T,J)
    """
    data_cfg = config["data"]
    seq_len = data_cfg["sequence_length"]
    grid_h, grid_w = data_cfg["pressure_grid_size"]
    n_joints = data_cfg["skeleton_joints"]

    # IMU: API input [T, 6]
    imu_np = np.array(data.imu, dtype=np.float32)
    imu_proc = preprocess_imu(imu_np, seq_len)           # (6, T)
    imu_t = torch.from_numpy(imu_proc).unsqueeze(0)      # (1, 6, T)

    # Pressure: API input [H, W] (single frame) → tile to [T, H, W]
    pressure_np = np.array(data.pressure, dtype=np.float32)
    if pressure_np.ndim == 2:
        pressure_np = np.stack([pressure_np] * seq_len, axis=0)  # (T, H, W)
    pressure_proc = preprocess_pressure(
        pressure_np, seq_len, (grid_h, grid_w)
    )                                                     # (T, 1, H, W)
    pressure_t = torch.from_numpy(pressure_proc).unsqueeze(0)  # (1, T, 1, H, W)

    # Skeleton: API input [T, J, 3]
    skeleton_np = np.array(data.skeleton, dtype=np.float32)
    skeleton_proc = preprocess_skeleton(skeleton_np, seq_len, n_joints)  # (3, T, J)
    skeleton_t = torch.from_numpy(skeleton_proc).unsqueeze(0)            # (1, 3, T, J)

    return {"imu": imu_t, "pressure": pressure_t, "skeleton": skeleton_t}


def _features_to_dict(features: GaitFeatures) -> dict:
    d = features.model_dump()
    # Fill optional biomarkers with derived defaults
    if d.get("ml_variability") is None:
        d["ml_variability"] = d["ml_index"] * 0.5
    if d.get("heel_pressure_ratio") is None:
        d["heel_pressure_ratio"] = (
            d["zone_heel_medial_mean"] + d["zone_heel_lateral_mean"]
        )
    if d.get("forefoot_pressure_ratio") is None:
        d["forefoot_pressure_ratio"] = (
            d["zone_forefoot_medial_mean"] + d["zone_forefoot_lateral_mean"]
        )
    if d.get("pressure_asymmetry") is None:
        d["pressure_asymmetry"] = abs(
            d["zone_heel_medial_mean"] - d["zone_heel_lateral_mean"]
        )
    if d.get("trunk_sway") is None:
        d["trunk_sway"] = d["cop_sway"] * 1.2
    return d


class GaitMLService:
    """Singleton service holding all loaded models."""

    def __init__(self):
        self._config = _load_config()
        self._device = torch.device("cpu")
        # Lazy-initialized neural network models (per checkpoint path)
        self._classify_models: dict[str | None, tuple[MultimodalGaitNet, bool]] = {}
        self._reasoning_models: dict[str | None, tuple[GaitReasoningEngine, bool]] = {}
        # Rule-based + sklearn models (trained on synthetic data at startup)
        self._disease_predictor = DiseaseRiskPredictor()
        self._disease_clf = GaitDiseaseClassifier(n_estimators=100)
        self._injury_predictor = InjuryRiskPredictor(n_estimators=100)
        self._sklearn_trained = False

    def warmup(self):
        """Pre-train sklearn models on synthetic data (called at startup)."""
        self._disease_clf.train()
        self._injury_predictor.train()
        self._sklearn_trained = True

    # ── Neural network loaders ──────────────────────────────────────────

    def _get_classify_model(self, ckpt: str | None) -> tuple[MultimodalGaitNet, bool]:
        if ckpt not in self._classify_models:
            model = MultimodalGaitNet(self._config).to(self._device)
            demo = True
            if ckpt and Path(ckpt).exists():
                state = torch.load(ckpt, map_location=self._device, weights_only=False)
                model.load_state_dict(state["model_state_dict"])
                demo = False
            model.eval()
            self._classify_models[ckpt] = (model, demo)
        return self._classify_models[ckpt]

    def _get_reasoning_model(self, ckpt: str | None) -> tuple[GaitReasoningEngine, bool]:
        if ckpt not in self._reasoning_models:
            model = GaitReasoningEngine(self._config).to(self._device)
            demo = True
            if ckpt and Path(ckpt).exists():
                model.load_base_model_weights(ckpt, device=self._device)
                demo = False
            model.eval()
            self._reasoning_models[ckpt] = (model, demo)
        return self._reasoning_models[ckpt]

    # ── Public inference methods ────────────────────────────────────────

    def classify(self, sensor_data: SensorData, ckpt: str | None = None) -> GaitClassifyResponse:
        model, is_demo = self._get_classify_model(ckpt)
        batch = _sensor_to_tensors(sensor_data, self._config)

        with torch.no_grad():
            logits = model(batch)  # [1, num_classes]
            probs = torch.softmax(logits, dim=-1)[0].numpy()

        pred_idx = int(probs.argmax())
        pred_en, pred_kr = GAIT_CLASS_NAMES[pred_idx]

        return GaitClassifyResponse(
            prediction=pred_en,
            prediction_kr=pred_kr,
            confidence=float(probs[pred_idx]),
            class_probabilities={
                GAIT_CLASS_NAMES[i][0]: float(probs[i])
                for i in range(len(probs))
            },
            is_demo_mode=is_demo,
        )

    def disease_risk(self, features: GaitFeatures) -> DiseaseRiskResponse:
        feat_dict = _features_to_dict(features)

        # Rule-based risk assessment
        report = self._disease_predictor.predict(feat_dict)

        # ML classification
        clf_result = self._disease_clf.predict(feat_dict)

        top_diseases = []
        for d in report.top_risks[:5]:
            top_diseases.append(DiseaseRisk(
                disease=d.disease_id,
                disease_kr=d.korean_name,
                risk_score=d.risk_score,
                severity=d.severity,
                key_signs=d.key_signs[:3],
                referral=d.referral,
            ))

        abnormal = [b.korean_name for b in report.biomarker_profile.biomarkers if b.is_abnormal]

        return DiseaseRiskResponse(
            top_diseases=top_diseases,
            ml_prediction=clf_result.predicted_class,
            ml_prediction_kr=clf_result.predicted_korean,
            ml_confidence=clf_result.confidence,
            ml_top3=[
                {"name_kr": name, "probability": round(prob, 4)}
                for name, prob in clf_result.top3
            ],
            abnormal_biomarkers=abnormal[:8],
        )

    def injury_risk(self, features: GaitFeatures) -> InjuryRiskResponse:
        feat_dict = _features_to_dict(features)
        report = self._injury_predictor.predict_comprehensive(feat_dict)
        ml = report.ml_prediction

        return InjuryRiskResponse(
            predicted_injury=ml.predicted_injury,
            predicted_injury_kr=ml.predicted_korean,
            confidence=ml.confidence,
            top3=[
                {"name_kr": name, "probability": round(prob, 4)}
                for name, prob in ml.top3
            ],
            body_risk_map=report.body_risk_map,
            timeline=ml.timeline,
            combined_risk_score=report.combined_risk_score,
            combined_risk_grade=report.combined_risk_grade,
            priority_actions=report.priority_actions[:5],
        )

    def reasoning(self, sensor_data: SensorData, ckpt: str | None = None) -> ReasoningResponse:
        import torch.nn.functional as F

        model, is_demo = self._get_reasoning_model(ckpt)
        batch = _sensor_to_tensors(sensor_data, self._config)

        result = model.reason(batch)

        pred_idx = int(result["prediction"][0].item())
        probs = result["calibrated_probs"][0].cpu().numpy()
        uncertainty = float(result["uncertainty"][0].item())
        pred_en, pred_kr = GAIT_CLASS_NAMES[pred_idx]

        # Anomaly findings
        anomaly_findings = []
        for m_name, anom in zip(MODALITY_NAMES, result["anomaly_results"]):
            scores = anom["anomaly_scores"][0].cpu().numpy()
            detected = [
                {"type": ANOMALY_NAMES[i], "score": round(float(s), 3)}
                for i, s in enumerate(scores) if s > 0.5
            ]
            anomaly_findings.append({"modality": m_name, "anomalies": detected})

        # Modality weights
        weights = result["evidence"]["modality_weights"][0].cpu().numpy()
        modality_weights = {
            name: round(float(w), 3)
            for name, w in zip(MODALITY_NAMES, weights)
        }

        # Reasoning trace
        trace = result["diagnosis"]["reasoning_trace"]
        reasoning_trace = []
        for step_idx, step_logits in enumerate(trace):
            step_probs = F.softmax(step_logits[0], dim=-1).cpu().numpy()
            top_cls = int(step_probs.argmax())
            label = "초기 가설" if step_idx == 0 else f"추론 {step_idx}단계"
            en, kr = GAIT_CLASS_NAMES[top_cls]
            reasoning_trace.append(ReasoningStep(
                step=step_idx,
                label=label,
                prediction=en,
                prediction_kr=kr,
                probability=round(float(step_probs[top_cls]), 3),
            ))

        report_kr = model.explain(result, sample_idx=0)

        return ReasoningResponse(
            final_prediction=pred_en,
            final_prediction_kr=pred_kr,
            confidence=round(float(probs[pred_idx]), 4),
            uncertainty=round(uncertainty, 4),
            anomaly_findings=anomaly_findings,
            modality_weights=modality_weights,
            evidence_strength=round(float(result["evidence"]["evidence_strength"][0].item()), 4),
            reasoning_trace=reasoning_trace,
            report_kr=report_kr,
            is_demo_mode=is_demo,
        )

    def analyze(
        self,
        sensor_data: SensorData,
        features: GaitFeatures,
        ckpt: str | None = None,
    ) -> AnalyzeResponse:
        return AnalyzeResponse(
            classify=self.classify(sensor_data, ckpt),
            disease_risk=self.disease_risk(features),
            injury_risk=self.injury_risk(features),
            reasoning=self.reasoning(sensor_data, ckpt),
        )


_service: GaitMLService | None = None


def get_service() -> GaitMLService:
    global _service
    if _service is None:
        _service = GaitMLService()
    return _service
