"""ML 기반 부상 위험 예측기.

비정상 보행 패턴 감지 결과와 보행 바이오마커를 결합하여
부상 위험을 머신러닝으로 예측합니다:
  - Random Forest 기반 부상 유형 분류
  - 합성 학습 데이터 (부상 시나리오별 보행 프로파일)
  - 부상 발생 시기 예측 (급성/만성)
  - 신체 부위별 위험 히트맵
"""

import numpy as np
from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from .gait_anomaly import GaitAnomalyDetector, GaitAnomalyReport


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

FEATURE_KOREAN = {
    "gait_speed": "보행 속도",
    "cadence": "보행률",
    "stride_regularity": "보폭 규칙성",
    "step_symmetry": "좌우 대칭성",
    "cop_sway": "체중심 흔들림",
    "ml_variability": "좌우 변동성",
    "heel_pressure_ratio": "뒤꿈치 하중",
    "forefoot_pressure_ratio": "앞발 하중",
    "arch_index": "아치 지수",
    "pressure_asymmetry": "압력 비대칭",
    "acceleration_rms": "가속도 크기",
    "trunk_sway": "체간 흔들림",
    "anomaly_score": "이상 점수",
}


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


@dataclass
class InjuryPredictorMetrics:
    """예측기 학습 성과."""
    accuracy: float
    f1_macro: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    feature_importance: dict[str, float]


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


class InjuryRiskPredictor:
    """ML 기반 부상 위험 예측기.

    비정상 보행 패턴 감지 결과 + 바이오마커 → 부상 유형 분류.
    합성 데이터로 사전 학습하고, 실 데이터로 파인튜닝 가능.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self.anomaly_detector = GaitAnomalyDetector()
        self.is_trained = False
        self._feature_importance = {}
        self.classes_ = []

    def generate_training_data(
        self,
        n_per_class: int = 150,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """부상 시나리오별 합성 학습 데이터를 생성합니다.

        각 부상 유형의 보행 특성 패턴을 반영한 가우시안 분포에서 샘플링.
        """
        rng = np.random.RandomState(seed)

        # 부상 유형별 보행 특성 프로파일 (mean, std)
        # [gait_speed, cadence, stride_reg, step_sym, cop_sway, ml_var,
        #  heel_ratio, fore_ratio, arch_idx, press_asym, accel_rms,
        #  trunk_sway, anomaly_score]
        injury_profiles = {
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

        X_all, y_all = [], []
        for label, profile in injury_profiles.items():
            mean = np.array(profile["mean"])
            std = np.array(profile["std"])
            samples = rng.normal(loc=mean, scale=std, size=(n_per_class, len(mean)))
            samples = np.clip(samples, 0, None)
            X_all.append(samples)
            y_all.append(np.full(n_per_class, label))

        X = np.vstack(X_all).astype(np.float32)
        y = np.concatenate(y_all).astype(np.int64)

        idx = rng.permutation(len(y))
        return X[idx], y[idx]

    def train(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        cv_folds: int = 5,
    ) -> InjuryPredictorMetrics:
        """예측기 학습 및 교차 검증."""
        if X is None or y is None:
            X, y = self.generate_training_data()

        self.classes_ = sorted(set(y.tolist()))
        X_scaled = self.scaler.fit_transform(X)

        # 교차 검증
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in cv.split(X_scaled, y):
            self.rf.fit(X_scaled[train_idx], y[train_idx])
            pred = self.rf.predict(X_scaled[val_idx])
            cv_scores.append(accuracy_score(y[val_idx], pred))

        # 전체 데이터로 최종 학습
        self.rf.fit(X_scaled, y)
        self.is_trained = True

        # 특성 중요도
        importances = self.rf.feature_importances_
        self._feature_importance = {
            PREDICTOR_FEATURES[i]: float(importances[i])
            for i in range(min(len(PREDICTOR_FEATURES), len(importances)))
        }

        y_pred = self.rf.predict(X_scaled)

        return InjuryPredictorMetrics(
            accuracy=accuracy_score(y, y_pred),
            f1_macro=f1_score(y, y_pred, average="macro", zero_division=0),
            cv_accuracy_mean=float(np.mean(cv_scores)),
            cv_accuracy_std=float(np.std(cv_scores)),
            feature_importance=self._feature_importance,
        )

    def predict(self, features: dict[str, float]) -> InjuryPrediction:
        """보행 특성으로 부상 위험을 예측합니다."""
        if not self.is_trained:
            self.train()

        # anomaly_score 계산
        anomaly_report = self.anomaly_detector.detect(features)
        enriched = dict(features)
        enriched["anomaly_score"] = anomaly_report.anomaly_score

        # 파생 특성 계산
        self.anomaly_detector._compute_derived(enriched)

        # 특성 벡터 구성
        x = np.array([enriched.get(f, 0.0) for f in PREDICTOR_FEATURES]).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        # 예측
        proba = self.rf.predict_proba(x_scaled)[0]
        pred_idx = int(np.argmax(proba))
        pred_class = self.classes_[pred_idx]
        pred_id, pred_korean = INJURY_LABELS[pred_class]

        # 확률 매핑
        probabilities = {}
        for i, cls in enumerate(self.classes_):
            _, label_kr = INJURY_LABELS[cls]
            probabilities[label_kr] = float(proba[i])

        # Top 3
        top_indices = np.argsort(proba)[::-1][:3]
        top3 = [
            (INJURY_LABELS[self.classes_[i]][1], float(proba[i]))
            for i in top_indices
        ]

        # 신체 부위 위험 맵
        body_risk_map = self._compute_body_risk_map(probabilities)

        # 타임라인
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

        # 1. 비정상 패턴 감지
        anomaly_report = self.anomaly_detector.detect(features)

        # 2. ML 예측
        ml_prediction = self.predict(features)

        # 3. 종합 위험 점수 (패턴 감지 40% + ML 예측 60%)
        ml_risk = 1.0 - ml_prediction.probabilities.get("낮은 위험", 0.0)
        combined_risk = 0.4 * anomaly_report.anomaly_score + 0.6 * ml_risk
        combined_risk = float(np.clip(combined_risk, 0, 1))

        combined_grade = self._risk_grade(combined_risk)

        # 4. 신체 부위 종합 위험도
        body_map = {}
        # ML 기반
        for part, score in ml_prediction.body_risk_map.items():
            body_map[part] = body_map.get(part, 0) + score * 0.6
        # 패턴 기반
        for injury, score in anomaly_report.injury_risk_summary.items():
            from .gait_anomaly import INJURY_CATEGORIES
            cat = INJURY_CATEGORIES.get(injury, {})
            part = cat.get("body_part", "")
            if part:
                body_map[part] = body_map.get(part, 0) + score * 0.4
        # 정규화
        if body_map:
            max_val = max(body_map.values()) + 1e-8
            body_map = {k: round(min(v / max_val, 1.0), 3) for k, v in body_map.items()}
        body_map = dict(sorted(body_map.items(), key=lambda x: -x[1]))

        # 5. 우선 조치 사항
        priority_actions = self._generate_priority_actions(
            anomaly_report, ml_prediction, combined_risk
        )

        # 6. 종합 보고서
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
            # 한국어 → 영어 ID 변환
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

        # 정규화
        if body_risk:
            max_val = max(body_risk.values()) + 1e-8
            body_risk = {k: round(min(v / max_val, 1.0), 3) for k, v in body_risk.items()}

        return dict(sorted(body_risk.items(), key=lambda x: -x[1]))

    def _risk_grade(self, score: float) -> str:
        if score >= 0.75:
            return "위험"
        elif score >= 0.50:
            return "경고"
        elif score >= 0.25:
            return "주의"
        elif score > 0.05:
            return "경미"
        else:
            return "정상"

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

        # 상위 이상 패턴 교정
        for p in anomaly_report.abnormal_patterns[:2]:
            if p.correction:
                actions.append(f"교정 필요: {p.korean_name} → {p.correction}")

        # 신체 부위 주의
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
            "=" * 65,
            "  슈올즈 AI — 종합 부상 위험 예측 보고서",
            "  (비정상 보행 패턴 감지 + ML 부상 예측)",
            "=" * 65,
            "",
        ]

        # 종합 위험도
        lines.append(f"  종합 부상 위험: {combined_risk:.0%} ({combined_grade})")
        lines.append(f"  이상 패턴 감지: {len(anomaly_report.abnormal_patterns)}개")
        lines.append(
            f"  ML 예측 부상: {ml_prediction.predicted_korean} "
            f"(신뢰도 {ml_prediction.confidence:.0%})"
        )
        lines.append("")

        # 비정상 패턴 요약
        lines.append("─" * 65)
        lines.append("  [감지된 비정상 보행 패턴]")
        lines.append("")

        if anomaly_report.abnormal_patterns:
            for p in anomaly_report.abnormal_patterns[:6]:
                bar_len = int(p.severity * 15)
                bar = "█" * bar_len + "░" * (15 - bar_len)
                lines.append(
                    f"  ▲ {p.korean_name:16s} [{bar}] "
                    f"{p.severity:.0%} ({p.severity_label})"
                )
        else:
            lines.append("  ○ 유의미한 이상 패턴이 감지되지 않았습니다.")
        lines.append("")

        # ML 부상 예측 Top3
        lines.append("─" * 65)
        lines.append("  [ML 부상 위험 예측]")
        lines.append("")
        for rank, (name, prob) in enumerate(ml_prediction.top3, 1):
            bar_len = int(prob * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {rank}. {name:14s} [{bar}] {prob:.1%}")
        lines.append(f"  예상 발생 시기: {ml_prediction.timeline}")
        lines.append("")

        # 신체 부위 위험 맵
        if body_map:
            lines.append("─" * 65)
            lines.append("  [신체 부위별 위험도]")
            lines.append("")
            for part, score in list(body_map.items())[:6]:
                bar_len = int(score * 15)
                bar = "█" * bar_len + "░" * (15 - bar_len)
                grade = self._risk_grade(score)
                lines.append(f"  {part:10s} [{bar}] {score:.0%} ({grade})")
            lines.append("")

        # 우선 조치
        lines.append("─" * 65)
        lines.append("  [우선 조치 사항]")
        lines.append("")
        for i, action in enumerate(priority_actions, 1):
            lines.append(f"  {i}. {action}")

        lines.append("")
        lines.append("  ※ 본 결과는 AI 기반 스크리닝이며, 확진은 전문의 상담이 필요합니다.")
        lines.append("=" * 65)
        return "\n".join(lines)
