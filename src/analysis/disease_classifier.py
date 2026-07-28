"""ML 기반 질환 분류기.

보행 바이오마커 벡터로부터 질환을 분류하는 머신러닝 모델:
  - BaseGaitClassifier 기반 (Random Forest)
  - Gradient Boosting 앙상블 추가
  - 교차 검증 + 특성 중요도 분석

11개 질환 클래스:
  - 신경계: 정상, 파킨슨병, 치매(알츠하이머), 소뇌 실조증
  - 뇌혈관계: 뇌졸중, 뇌출혈, 뇌경색
  - 근골격계: 골관절염, 류마티스 관절염, 추간판 탈출증(디스크)
  - 대사/신경: 당뇨 신경병증
"""

import numpy as np
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

from .base_classifier import BaseGaitClassifier, TrainingMetrics
from .common import get_feature_korean


@dataclass
class ClassificationResult:
    """분류 결과."""
    predicted_class: str
    predicted_korean: str
    probabilities: dict[str, float]    # class -> probability
    top3: list[tuple[str, float]]      # [(class, prob), ...]
    confidence: float
    feature_importance: dict[str, float]


@dataclass
class ClassifierMetrics(TrainingMetrics):
    """분류기 학습 성과 (TrainingMetrics + per_class_report)."""
    per_class_report: str = ""


# 질환 라벨 매핑
DISEASE_LABELS = {
    0: ("normal", "정상 보행"),
    1: ("parkinsons", "파킨슨병"),
    2: ("stroke", "뇌졸중"),
    3: ("diabetic_neuropathy", "당뇨 신경병증"),
    4: ("cerebellar_ataxia", "소뇌 실조증"),
    5: ("osteoarthritis", "골관절염"),
    6: ("dementia", "치매 (알츠하이머)"),
    7: ("cerebral_hemorrhage", "뇌출혈"),
    8: ("cerebral_infarction", "뇌경색"),
    9: ("disc_herniation", "추간판 탈출증"),
    10: ("rheumatoid_arthritis", "류마티스 관절염"),
}

# 특성 이름 (고정 순서)
FEATURE_NAMES = [
    "gait_speed", "cadence", "stride_regularity", "step_symmetry",
    "cop_sway", "ml_variability", "heel_pressure_ratio", "forefoot_pressure_ratio",
    "arch_index", "pressure_asymmetry", "acceleration_rms",
    "acceleration_variability", "trunk_sway",
]

# FEATURE_KOREAN은 common.py에서 import (하위 호환용 재export)
FEATURE_KOREAN = {f: get_feature_korean(f) for f in FEATURE_NAMES}

# ── 질환별 합성 데이터 프로파일 ────────────────────────────────────────
_DISEASE_PROFILES = {
    0: {  # 정상
        "mean": [1.2, 115, 0.85, 0.92, 0.04, 0.06, 0.32, 0.45, 0.25, 0.05, 1.5, 0.15, 2.0],
        "std":  [0.15, 10, 0.06, 0.04, 0.015, 0.02, 0.04, 0.05, 0.05, 0.03, 0.3, 0.05, 0.5],
    },
    1: {  # 파킨슨
        "mean": [0.7, 145, 0.50, 0.80, 0.06, 0.09, 0.30, 0.48, 0.24, 0.08, 1.0, 0.35, 2.8],
        "std":  [0.15, 15, 0.10, 0.08, 0.02, 0.03, 0.04, 0.05, 0.04, 0.04, 0.3, 0.08, 0.6],
    },
    2: {  # 뇌졸중
        "mean": [0.6, 90, 0.65, 0.60, 0.07, 0.14, 0.28, 0.50, 0.26, 0.20, 1.2, 0.25, 3.0],
        "std":  [0.15, 12, 0.08, 0.10, 0.02, 0.04, 0.05, 0.06, 0.05, 0.06, 0.3, 0.06, 0.7],
    },
    3: {  # 당뇨 신경병증
        "mean": [0.9, 105, 0.60, 0.85, 0.08, 0.10, 0.20, 0.60, 0.38, 0.08, 1.3, 0.22, 2.5],
        "std":  [0.12, 10, 0.08, 0.05, 0.025, 0.03, 0.04, 0.06, 0.06, 0.04, 0.25, 0.05, 0.5],
    },
    4: {  # 소뇌 실조
        "mean": [0.8, 100, 0.45, 0.78, 0.10, 0.16, 0.30, 0.46, 0.27, 0.10, 1.4, 0.38, 4.0],
        "std":  [0.15, 12, 0.10, 0.08, 0.03, 0.04, 0.05, 0.06, 0.05, 0.04, 0.3, 0.08, 0.8],
    },
    5: {  # 골관절염
        "mean": [0.85, 100, 0.72, 0.75, 0.05, 0.08, 0.38, 0.42, 0.26, 0.15, 1.3, 0.20, 2.3],
        "std":  [0.12, 10, 0.07, 0.08, 0.02, 0.03, 0.05, 0.05, 0.04, 0.05, 0.25, 0.05, 0.5],
    },
    6: {  # 치매 (알츠하이머)
        "mean": [0.7, 90, 0.55, 0.82, 0.06, 0.08, 0.33, 0.44, 0.25, 0.07, 1.1, 0.32, 2.6],
        "std":  [0.15, 12, 0.10, 0.06, 0.02, 0.03, 0.04, 0.05, 0.05, 0.04, 0.3, 0.07, 0.6],
    },
    7: {  # 뇌출혈
        "mean": [0.50, 85, 0.50, 0.55, 0.08, 0.18, 0.25, 0.52, 0.27, 0.22, 1.1, 0.30, 3.5],
        "std":  [0.15, 12, 0.10, 0.12, 0.025, 0.05, 0.05, 0.06, 0.05, 0.07, 0.3, 0.07, 0.8],
    },
    8: {  # 뇌경색
        "mean": [0.70, 95, 0.60, 0.68, 0.065, 0.12, 0.26, 0.50, 0.26, 0.15, 1.2, 0.25, 2.8],
        "std":  [0.12, 10, 0.08, 0.08, 0.02, 0.04, 0.04, 0.05, 0.04, 0.05, 0.25, 0.06, 0.6],
    },
    9: {  # 추간판 탈출증 (디스크)
        "mean": [0.85, 105, 0.68, 0.78, 0.055, 0.08, 0.22, 0.48, 0.26, 0.12, 0.95, 0.20, 3.0],
        "std":  [0.12, 10, 0.08, 0.07, 0.02, 0.03, 0.04, 0.05, 0.04, 0.04, 0.25, 0.05, 0.6],
    },
    10: {  # 류마티스 관절염
        "mean": [0.80, 100, 0.70, 0.80, 0.06, 0.07, 0.28, 0.58, 0.36, 0.10, 1.2, 0.18, 2.4],
        "std":  [0.12, 10, 0.07, 0.06, 0.02, 0.03, 0.04, 0.06, 0.06, 0.04, 0.25, 0.05, 0.5],
    },
}


class GaitDiseaseClassifier(BaseGaitClassifier):
    """보행 바이오마커 기반 질환 분류기.

    BaseGaitClassifier + Gradient Boosting 앙상블.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__(n_estimators=n_estimators, random_state=random_state)
        # RF max_depth을 disease classifier 원래 값으로 오버라이드
        self.rf.set_params(max_depth=10, min_samples_leaf=3)
        self.gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
        )

    @property
    def labels(self) -> dict[int, tuple[str, str]]:
        return DISEASE_LABELS

    @property
    def feature_names(self) -> list[str]:
        return FEATURE_NAMES

    def _get_profiles(self) -> dict[int, dict]:
        return _DISEASE_PROFILES

    def _on_train_complete(self, X_scaled: np.ndarray, y: np.ndarray) -> None:
        """GB 앙상블 추가 학습."""
        self.gb.fit(X_scaled, y)
        # 특성 중요도를 RF+GB 평균으로 업데이트
        rf_imp = self.rf.feature_importances_
        gb_imp = self.gb.feature_importances_
        avg_imp = (rf_imp + gb_imp) / 2
        self._feature_importance = {
            self.feature_names[i]: float(avg_imp[i])
            for i in range(min(len(self.feature_names), len(avg_imp)))
        }

    def _predict_proba(self, x_scaled: np.ndarray) -> np.ndarray:
        """RF + GB 앙상블 예측."""
        rf_proba = self.rf.predict_proba(x_scaled)[0]
        gb_proba = self.gb.predict_proba(x_scaled)[0]
        return (rf_proba + gb_proba) / 2

    def train(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        cv_folds: int = 5,
    ) -> ClassifierMetrics:
        """학습 + per_class_report 포함 결과 반환."""
        base_metrics = super().train(X, y, cv_folds)

        # per_class_report 생성
        if X is None or y is None:
            X, y = self.generate_training_data()
        X_scaled = self.scaler.transform(X)
        y_pred = self.rf.predict(X_scaled)
        report = classification_report(
            y, y_pred,
            target_names=[DISEASE_LABELS[c][1] for c in self.classes_],
            zero_division=0,
        )

        return ClassifierMetrics(
            accuracy=base_metrics.accuracy,
            f1_macro=base_metrics.f1_macro,
            cv_accuracy_mean=base_metrics.cv_accuracy_mean,
            cv_accuracy_std=base_metrics.cv_accuracy_std,
            feature_importance=base_metrics.feature_importance,
            per_class_report=report,
        )

    def predict(self, features: dict[str, float]) -> ClassificationResult:
        """보행 특성으로 질환 분류."""
        proba, pred_idx, pred_class = self._predict_base(features)
        pred_id, pred_korean = self.labels[pred_class]

        return ClassificationResult(
            predicted_class=pred_id,
            predicted_korean=pred_korean,
            probabilities=self._build_probabilities(proba),
            top3=self._build_top3(proba),
            confidence=float(proba[pred_idx]),
            feature_importance=self._feature_importance,
        )

    def get_feature_importance_report(self) -> str:
        """특성 중요도 한국어 보고서."""
        if not self._feature_importance:
            return "학습이 필요합니다."

        sorted_feats = sorted(
            self._feature_importance.items(), key=lambda x: -x[1]
        )

        lines = ["  [특성 중요도 순위]", ""]
        for rank, (feat, imp) in enumerate(sorted_feats, 1):
            kr_name = get_feature_korean(feat)
            bar = "█" * int(imp * 50) + "░" * (10 - int(imp * 50))
            lines.append(f"  {rank:2d}. {kr_name:12s} [{bar}] {imp:.3f}")

        return "\n".join(lines)
