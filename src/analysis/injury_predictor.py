"""ML 기반 부상 위험 예측기.

비정상 보행 패턴 감지 결과와 보행 바이오마커를 결합하여
부상 위험을 머신러닝으로 예측합니다:
  - BaseGaitClassifier 기반 부상 유형 분류
  - 합성 학습 데이터 (부상 시나리오별 보행 프로파일)
  - 부상 발생 시기 예측 (급성/만성)
  - 신체 부위별 위험 히트맵
"""

import numpy as np
from dataclasses import dataclass, field

from .base_classifier import BaseGaitClassifier, TrainingMetrics
from .common import severity_label, compute_derived_features
from .report_formatter import header, section, marker_line, risk_line, ranked_line, HEADER_DIVIDER
from .gait_anomaly import GaitAnomalyDetector, GaitAnomalyReport, INJURY_CATEGORIES


# ── 부상 유형 정의 ────────────────────────────────────────────────────
INJURY_LABELS = {
    0: ("low_risk", "낮은 위험"),
    1: ("plantar_fasciitis", "족저근막염"),
    2: ("metatarsal_stress", "중족골 피로골절"),
    3: ("ankle_sprain", "발목 염좌"),
    4: ("knee_overload", "무릎 과부하"),
    5: ("hip_back_pain", "고관절/요통"),
    6: ("fall_risk", "낙상 위험"),
    7: ("achilles_tendinitis", "아킬레스건염"),
    8: ("shin_splint", "경골 스트레스"),
}

PREDICTOR_FEATURES = [
    "gait_speed", "cadence", "stride_regularity", "step_symmetry",
    "cop_sway", "ml_variability", "heel_pressure_ratio",
    "forefoot_pressure_ratio", "arch_index", "pressure_asymmetry",
    "acceleration_rms", "trunk_sway",
    "anomaly_score",
]


@dataclass
class InjuryPrediction:
    """부상 예측 결과."""
    predicted_injury: str
    predicted_korean: str
    confidence: float
    probabilities: dict[str, float]     # 부상유형 → 확률
    top3: list[tuple[str, float]]       # [(한국어명, 확률), ...]
    body_risk_map: dict[str, float]     # 신체부위 → 위험도
    timeline: str                       # 급성/만성/복합
    feature_importance: dict[str, float]


# InjuryPredictorMetrics는 TrainingMetrics로 대체 (하위 호환용 alias)
InjuryPredictorMetrics = TrainingMetrics


@dataclass
class ComprehensiveInjuryReport:
    """종합 부상 위험 보고서 (패턴 감지 + ML 예측)."""
    anomaly_report: GaitAnomalyReport
    ml_prediction: InjuryPrediction
    combined_risk_score: float       # 0~1
    combined_risk_grade: str         # 정상/경미/주의/경고/위험
    body_risk_map: dict[str, float]  # 신체부위별 종합 위험도
    priority_actions: list[str]      # 우선 조치 사항
    summary_kr: str


# ── 부상 시나리오별 신체 부위 매핑 ─────────────────────────────────────
BODY_PART_MAP = {
    "low_risk": {},
    "plantar_fasciitis": {"발바닥": 0.9, "뒤꿈치": 0.7, "종아리": 0.3},
    "metatarsal_stress": {"전족부": 0.9, "발등": 0.5, "발가락": 0.4},
    "ankle_sprain": {"발목": 0.9, "발 외측": 0.6, "종아리": 0.3},
    "knee_overload": {"무릎": 0.9, "대퇴부": 0.5, "정강이": 0.4},
    "hip_back_pain": {"고관절": 0.8, "허리": 0.8, "대퇴부": 0.4},
    "fall_risk": {"전신": 0.9, "손목": 0.6, "고관절": 0.7, "머리": 0.5},
    "achilles_tendinitis": {"아킬레스건": 0.9, "종아리": 0.7, "뒤꿈치": 0.5},
    "shin_splint": {"정강이": 0.9, "발목": 0.4, "무릎": 0.3},
}

TIMELINE_MAP = {
    "low_risk": "해당 없음",
    "plantar_fasciitis": "만성 (2~6주 내 증상 발현)",
    "metatarsal_stress": "급성/만성 (4~8주 내 골절 위험)",
    "ankle_sprain": "급성 (즉각적 부상 위험)",
    "knee_overload": "만성 (수주~수개월 축적)",
    "hip_back_pain": "만성 (보상 패턴 축적)",
    "fall_risk": "급성 (즉각적 낙상 위험)",
    "achilles_tendinitis": "만성 (2~4주 내 증상 발현)",
    "shin_splint": "만성 (과도한 훈련 시 2~4주)",
}

# ── 부상별 합성 데이터 프로파일 ────────────────────────────────────────
_INJURY_PROFILES = {
    0: {  # 낮은 위험 (정상)
        "mean": [1.2, 115, 0.85, 0.92, 0.04, 0.06, 0.32, 0.45, 0.25, 0.05, 1.5, 2.0, 0.05],
        "std":  [0.12, 8, 0.05, 0.03, 0.01, 0.02, 0.04, 0.04, 0.04, 0.02, 0.25, 0.4, 0.03],
    },
    1: {  # 족저근막염
        "mean": [1.0, 108, 0.78, 0.88, 0.055, 0.07, 0.42, 0.38, 0.18, 0.06, 1.8, 2.2, 0.35],
        "std":  [0.12, 10, 0.06, 0.04, 0.015, 0.02, 0.05, 0.04, 0.04, 0.03, 0.3, 0.4, 0.10],
    },
    2: {  # 중족골 피로골절
        "mean": [1.05, 112, 0.75, 0.85, 0.05, 0.07, 0.20, 0.65, 0.28, 0.07, 1.6, 2.1, 0.45],
        "std":  [0.12, 10, 0.07, 0.05, 0.015, 0.02, 0.04, 0.06, 0.05, 0.03, 0.3, 0.4, 0.12],
    },
    3: {  # 발목 염좌
        "mean": [1.0, 110, 0.72, 0.80, 0.07, 0.14, 0.30, 0.46, 0.30, 0.12, 1.4, 2.8, 0.50],
        "std":  [0.15, 12, 0.08, 0.07, 0.02, 0.04, 0.04, 0.05, 0.05, 0.04, 0.3, 0.5, 0.12],
    },
    4: {  # 무릎 과부하
        "mean": [0.95, 105, 0.74, 0.82, 0.055, 0.08, 0.40, 0.42, 0.26, 0.10, 2.2, 2.5, 0.40],
        "std":  [0.12, 10, 0.06, 0.05, 0.015, 0.03, 0.05, 0.05, 0.04, 0.04, 0.35, 0.5, 0.10],
    },
    5: {  # 고관절/요통
        "mean": [0.90, 100, 0.70, 0.78, 0.06, 0.09, 0.28, 0.48, 0.27, 0.14, 1.3, 3.2, 0.48],
        "std":  [0.12, 10, 0.08, 0.06, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05, 0.25, 0.6, 0.12],
    },
    6: {  # 낙상 위험
        "mean": [0.70, 88, 0.55, 0.75, 0.09, 0.15, 0.30, 0.44, 0.28, 0.08, 1.0, 4.0, 0.65],
        "std":  [0.15, 12, 0.10, 0.08, 0.025, 0.04, 0.05, 0.05, 0.05, 0.04, 0.3, 0.7, 0.12],
    },
    7: {  # 아킬레스건염
        "mean": [1.0, 118, 0.78, 0.88, 0.05, 0.07, 0.38, 0.40, 0.22, 0.06, 1.7, 2.3, 0.35],
        "std":  [0.12, 10, 0.06, 0.04, 0.015, 0.02, 0.05, 0.05, 0.04, 0.03, 0.3, 0.4, 0.10],
    },
    8: {  # 경골 스트레스
        "mean": [1.05, 125, 0.76, 0.86, 0.05, 0.08, 0.40, 0.44, 0.24, 0.07, 2.0, 2.4, 0.38],
        "std":  [0.12, 12, 0.07, 0.04, 0.015, 0.02, 0.05, 0.05, 0.04, 0.03, 0.35, 0.4, 0.10],
    },
}


class InjuryRiskPredictor(BaseGaitClassifier):
    """ML 기반 부상 위험 예측기.

    BaseGaitClassifier를 상속하여 학습/예측 파이프라인을 재사용.
    비정상 보행 패턴 감지 결과 + 바이오마커 → 부상 유형 분류.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__(n_estimators=n_estimators, random_state=random_state)
        self.anomaly_detector = GaitAnomalyDetector()

    @property
    def labels(self) -> dict[int, tuple[str, str]]:
        return INJURY_LABELS

    @property
    def feature_names(self) -> list[str]:
        return PREDICTOR_FEATURES

    def _get_profiles(self) -> dict[int, dict]:
        return _INJURY_PROFILES

    def generate_training_data(self, n_per_class: int = 150, seed: int = 42):
        return super().generate_training_data(n_per_class=n_per_class, seed=seed)

    def predict(self, features: dict[str, float]) -> InjuryPrediction:
        """보행 특성으로 부상 위험을 예측합니다."""
        # anomaly_score 계산
        anomaly_report = self.anomaly_detector.detect(features)
        enriched = dict(features)
        enriched["anomaly_score"] = anomaly_report.anomaly_score
        compute_derived_features(enriched)

        proba, pred_idx, pred_class = self._predict_base(enriched)
        pred_id, pred_korean = self.labels[pred_class]

        probabilities = self._build_probabilities(proba)
        top3 = self._build_top3(proba)
        body_risk_map = self._compute_body_risk_map(probabilities)
        timeline = TIMELINE_MAP.get(pred_id, "")

        return InjuryPrediction(
            predicted_injury=pred_id,
            predicted_korean=pred_korean,
            confidence=float(proba[pred_idx]),
            probabilities=probabilities,
            top3=top3,
            body_risk_map=body_risk_map,
            timeline=timeline,
            feature_importance=self._feature_importance,
        )

    def predict_comprehensive(
        self,
        features: dict[str, float],
    ) -> ComprehensiveInjuryReport:
        """패턴 감지 + ML 예측을 결합한 종합 부상 위험 보고서."""
        if not self.is_trained:
            self.train()

        anomaly_report = self.anomaly_detector.detect(features)
        ml_prediction = self.predict(features)

        # 종합 위험 점수 (패턴 감지 40% + ML 예측 60%)
        ml_risk = 1.0 - ml_prediction.probabilities.get("낮은 위험", 0.0)
        combined_risk = float(np.clip(
            0.4 * anomaly_report.anomaly_score + 0.6 * ml_risk, 0, 1
        ))
        combined_grade = severity_label(combined_risk)

        # 신체 부위 종합 위험도
        body_map = self._combine_body_map(
            ml_prediction.body_risk_map, anomaly_report.injury_risk_summary
        )

        priority_actions = self._generate_priority_actions(
            anomaly_report, ml_prediction, combined_risk
        )

        summary = self._generate_comprehensive_summary(
            anomaly_report, ml_prediction, combined_risk,
            combined_grade, body_map, priority_actions
        )

        return ComprehensiveInjuryReport(
            anomaly_report=anomaly_report,
            ml_prediction=ml_prediction,
            combined_risk_score=round(combined_risk, 3),
            combined_risk_grade=combined_grade,
            body_risk_map=body_map,
            priority_actions=priority_actions,
            summary_kr=summary,
        )

    def _compute_body_risk_map(
        self,
        probabilities: dict[str, float],
    ) -> dict[str, float]:
        """ML 예측 확률에서 신체 부위별 위험도 계산."""
        body_risk: dict[str, float] = {}

        for injury_kr, prob in probabilities.items():
            injury_id = None
            for idx, (eid, ekr) in INJURY_LABELS.items():
                if ekr == injury_kr:
                    injury_id = eid
                    break
            if not injury_id or injury_id == "low_risk":
                continue

            parts = BODY_PART_MAP.get(injury_id, {})
            for part, weight in parts.items():
                body_risk[part] = body_risk.get(part, 0) + prob * weight

        return self._normalize_map(body_risk)

    @staticmethod
    def _normalize_map(m: dict[str, float]) -> dict[str, float]:
        """딕셔너리 값을 0~1로 정규화하고 내림차순 정렬."""
        if not m:
            return {}
        max_val = max(m.values()) + 1e-8
        normalized = {k: round(min(v / max_val, 1.0), 3) for k, v in m.items()}
        return dict(sorted(normalized.items(), key=lambda x: -x[1]))

    def _combine_body_map(
        self,
        ml_map: dict[str, float],
        anomaly_risks: dict[str, float],
    ) -> dict[str, float]:
        """ML + 패턴 기반 신체 부위 위험도 결합."""
        body_map: dict[str, float] = {}
        for part, score in ml_map.items():
            body_map[part] = body_map.get(part, 0) + score * 0.6
        for injury, score in anomaly_risks.items():
            cat = INJURY_CATEGORIES.get(injury, {})
            part = cat.get("body_part", "")
            if part:
                body_map[part] = body_map.get(part, 0) + score * 0.4
        return self._normalize_map(body_map)

    def _generate_priority_actions(
        self,
        anomaly_report: GaitAnomalyReport,
        ml_prediction: InjuryPrediction,
        combined_risk: float,
    ) -> list[str]:
        """우선 조치 사항 생성."""
        actions = []

        if combined_risk >= 0.50:
            actions.append("전문가(정형외과/재활의학과) 상담을 권장합니다")

        if ml_prediction.predicted_injury != "low_risk":
            actions.append(
                f"주요 부상 위험: {ml_prediction.predicted_korean} "
                f"(신뢰도 {ml_prediction.confidence:.0%})"
            )
            actions.append(f"예상 시기: {ml_prediction.timeline}")

        for p in anomaly_report.abnormal_patterns[:2]:
            if p.correction:
                actions.append(f"교정 필요: {p.korean_name} → {p.correction}")

        high_risk_parts = [
            part for part, score in ml_prediction.body_risk_map.items()
            if score >= 0.5
        ]
        if high_risk_parts:
            actions.append(f"주의 부위: {', '.join(high_risk_parts[:3])}")

        if not actions:
            actions.append("현재 부상 위험이 낮습니다. 정기 모니터링을 계속하세요.")

        return actions

    def _generate_comprehensive_summary(
        self,
        anomaly_report: GaitAnomalyReport,
        ml_prediction: InjuryPrediction,
        combined_risk: float,
        combined_grade: str,
        body_map: dict[str, float],
        priority_actions: list[str],
    ) -> str:
        """종합 보고서 생성."""
        lines = [
            header("슈올즈 AI — 종합 부상 위험 예측 보고서"),
            "  (비정상 보행 패턴 감지 + ML 부상 예측)",
            "",
        ]

        lines.append(f"  종합 부상 위험: {combined_risk:.0%} ({combined_grade})")
        lines.append(f"  이상 패턴 감지: {len(anomaly_report.abnormal_patterns)}개")
        lines.append(
            f"  ML 예측 부상: {ml_prediction.predicted_korean} "
            f"(신뢰도 {ml_prediction.confidence:.0%})"
        )
        lines.append("")

        # 비정상 패턴 요약
        lines.append(section("감지된 비정상 보행 패턴"))
        lines.append("")
        if anomaly_report.abnormal_patterns:
            for p in anomaly_report.abnormal_patterns[:6]:
                lines.append(marker_line(p.korean_name, p.severity))
        else:
            lines.append("  ○ 유의미한 이상 패턴이 감지되지 않았습니다.")
        lines.append("")

        # ML 부상 예측 Top3
        lines.append(section("ML 부상 위험 예측"))
        lines.append("")
        for rank, (name, prob) in enumerate(ml_prediction.top3, 1):
            lines.append(ranked_line(rank, name, prob))
        lines.append(f"  예상 발생 시기: {ml_prediction.timeline}")
        lines.append("")

        # 신체 부위 위험 맵
        if body_map:
            lines.append(section("신체 부위별 위험도"))
            lines.append("")
            for part, score in list(body_map.items())[:6]:
                lines.append(risk_line(part, score))
            lines.append("")

        # 우선 조치
        lines.append(section("우선 조치 사항"))
        lines.append("")
        for i, action in enumerate(priority_actions, 1):
            lines.append(f"  {i}. {action}")

        lines.append("")
        lines.append("  ※ 본 결과는 AI 기반 스크리닝이며, 확진은 전문의 상담이 필요합니다.")
        lines.append(HEADER_DIVIDER)
        return "\n".join(lines)
