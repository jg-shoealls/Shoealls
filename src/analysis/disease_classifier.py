"""ML 기반 질환 분류기.

보행 바이오마커 벡터로부터 질환을 분류하는 머신러닝 모델:
  - Random Forest (주력)
  - Gradient Boosting (앙상블)
  - 교차 검증 + 특성 중요도 분석
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report


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
class ClassifierMetrics:
    """분류기 학습 성과."""
    accuracy: float
    f1_macro: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    per_class_report: str
    feature_importance: dict[str, float]


# 질환 라벨 매핑
DISEASE_LABELS = {
    0: ("normal", "정상 보행"),
    1: ("parkinsons", "파킨슨병"),
    2: ("stroke", "뇌졸중"),
    3: ("diabetic_neuropathy", "당뇨 신경병증"),
    4: ("cerebellar_ataxia", "소뇌 실조증"),
    5: ("osteoarthritis", "골관절염"),
    6: ("dementia", "치매"),
}

# 특성 이름 (고정 순서)
FEATURE_NAMES = [
    "gait_speed", "cadence", "stride_regularity", "step_symmetry",
    "cop_sway", "ml_variability", "heel_pressure_ratio", "forefoot_pressure_ratio",
    "arch_index", "pressure_asymmetry", "acceleration_rms",
    "acceleration_variability", "trunk_sway",
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
    "acceleration_variability": "가속도 변동성",
    "trunk_sway": "체간 흔들림",
}


class GaitDiseaseClassifier:
    """보행 바이오마커 기반 질환 분류기.

    Random Forest + Gradient Boosting 앙상블.
    합성 데이터로 사전 학습하고, 실 데이터로 파인튜닝 가능.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_leaf=3,
            random_state=random_state,
            class_weight="balanced",
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self._feature_importance = {}
        self.classes_ = []

    def generate_training_data(
        self,
        n_per_class: int = 100,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """질환별 특성 패턴으로 합성 학습 데이터를 생성합니다.

        각 질환의 의학적 특성을 반영한 가우시안 분포에서 샘플링.
        """
        rng = np.random.RandomState(seed)

        # 질환별 특성 분포 정의 (mean, std)
        # [gait_speed, cadence, stride_reg, step_sym, cop_sway, ml_var,
        #  heel_ratio, fore_ratio, arch_idx, press_asym, accel_rms,
        #  accel_var, trunk_sway]
        disease_profiles = {
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
            6: {  # 치매
                "mean": [0.7, 90, 0.55, 0.82, 0.06, 0.08, 0.33, 0.44, 0.25, 0.07, 1.1, 0.32, 2.6],
                "std":  [0.15, 12, 0.10, 0.06, 0.02, 0.03, 0.04, 0.05, 0.05, 0.04, 0.3, 0.07, 0.6],
            },
        }

        X_all, y_all = [], []
        for label, profile in disease_profiles.items():
            mean = np.array(profile["mean"])
            std = np.array(profile["std"])
            samples = rng.normal(loc=mean, scale=std, size=(n_per_class, len(mean)))
            # 범위 제한
            samples = np.clip(samples, 0, None)
            X_all.append(samples)
            y_all.append(np.full(n_per_class, label))

        X = np.vstack(X_all).astype(np.float32)
        y = np.concatenate(y_all).astype(np.int64)

        # 셔플
        idx = rng.permutation(len(y))
        return X[idx], y[idx]

    def train(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        cv_folds: int = 5,
    ) -> ClassifierMetrics:
        """분류기 학습 및 교차 검증.

        X, y가 None이면 합성 데이터로 자동 학습.
        """
        if X is None or y is None:
            X, y = self.generate_training_data()

        self.classes_ = sorted(set(y.tolist()))

        # 스케일링
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
        self.gb.fit(X_scaled, y)
        self.is_trained = True

        # 특성 중요도
        rf_imp = self.rf.feature_importances_
        gb_imp = self.gb.feature_importances_
        avg_imp = (rf_imp + gb_imp) / 2

        self._feature_importance = {
            FEATURE_NAMES[i]: float(avg_imp[i])
            for i in range(min(len(FEATURE_NAMES), len(avg_imp)))
        }

        # 최종 성능
        y_pred = self.rf.predict(X_scaled)
        report = classification_report(
            y, y_pred,
            target_names=[DISEASE_LABELS[c][1] for c in self.classes_],
            zero_division=0,
        )

        return ClassifierMetrics(
            accuracy=accuracy_score(y, y_pred),
            f1_macro=f1_score(y, y_pred, average="macro", zero_division=0),
            cv_accuracy_mean=float(np.mean(cv_scores)),
            cv_accuracy_std=float(np.std(cv_scores)),
            per_class_report=report,
            feature_importance=self._feature_importance,
        )

    def predict(self, features: dict[str, float]) -> ClassificationResult:
        """보행 특성으로 질환 분류.

        Args:
            features: 바이오마커 딕셔너리.
        """
        if not self.is_trained:
            self.train()

        # 특성 벡터 구성
        x = np.array([features.get(f, 0.0) for f in FEATURE_NAMES]).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        # 앙상블 예측 (RF + GB 평균)
        rf_proba = self.rf.predict_proba(x_scaled)[0]
        gb_proba = self.gb.predict_proba(x_scaled)[0]
        avg_proba = (rf_proba + gb_proba) / 2

        pred_idx = int(np.argmax(avg_proba))
        pred_class = self.classes_[pred_idx]
        pred_id, pred_korean = DISEASE_LABELS[pred_class]

        # 확률 매핑
        probabilities = {}
        for i, cls in enumerate(self.classes_):
            label_id, label_kr = DISEASE_LABELS[cls]
            probabilities[label_kr] = float(avg_proba[i])

        # Top 3
        top_indices = np.argsort(avg_proba)[::-1][:3]
        top3 = [
            (DISEASE_LABELS[self.classes_[i]][1], float(avg_proba[i]))
            for i in top_indices
        ]

        return ClassificationResult(
            predicted_class=pred_id,
            predicted_korean=pred_korean,
            probabilities=probabilities,
            top3=top3,
            confidence=float(avg_proba[pred_idx]),
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
            kr_name = FEATURE_KOREAN.get(feat, feat)
            bar = "█" * int(imp * 50) + "░" * (10 - int(imp * 50))
            lines.append(f"  {rank:2d}. {kr_name:12s} [{bar}] {imp:.3f}")

        return "\n".join(lines)
