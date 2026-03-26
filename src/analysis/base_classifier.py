"""ML 분류기 공통 베이스 클래스.

GaitDiseaseClassifier와 InjuryRiskPredictor의 공통 패턴을 추출:
  - StandardScaler + RandomForest 학습 파이프라인
  - 합성 데이터 생성 (가우시안 프로파일)
  - 교차 검증 + 특성 중요도
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class TrainingMetrics:
    """분류기 학습 성과 (공통)."""
    accuracy: float
    f1_macro: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    feature_importance: dict[str, float]


class BaseGaitClassifier(ABC):
    """보행 분류기 공통 베이스.

    서브클래스는 LABELS, FEATURE_NAMES, _get_profiles()를 구현합니다.
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
        self.is_trained = False
        self._feature_importance: dict[str, float] = {}
        self.classes_: list[int] = []

    @property
    @abstractmethod
    def labels(self) -> dict[int, tuple[str, str]]:
        """클래스 라벨 매핑: {0: ("id", "한국어명"), ...}."""

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """특성 이름 목록 (고정 순서)."""

    @abstractmethod
    def _get_profiles(self) -> dict[int, dict]:
        """클래스별 합성 데이터 프로파일: {label: {"mean": [...], "std": [...]}}."""

    def generate_training_data(
        self,
        n_per_class: int = 100,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """프로파일 기반 합성 학습 데이터 생성."""
        rng = np.random.RandomState(seed)
        profiles = self._get_profiles()

        X_all, y_all = [], []
        for label, profile in profiles.items():
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
    ) -> TrainingMetrics:
        """학습 및 교차 검증."""
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

        # 전체 학습
        self.rf.fit(X_scaled, y)
        self.is_trained = True
        self._on_train_complete(X_scaled, y)

        # 특성 중요도
        importances = self.rf.feature_importances_
        self._feature_importance = {
            self.feature_names[i]: float(importances[i])
            for i in range(min(len(self.feature_names), len(importances)))
        }

        y_pred = self.rf.predict(X_scaled)

        return TrainingMetrics(
            accuracy=accuracy_score(y, y_pred),
            f1_macro=f1_score(y, y_pred, average="macro", zero_division=0),
            cv_accuracy_mean=float(np.mean(cv_scores)),
            cv_accuracy_std=float(np.std(cv_scores)),
            feature_importance=self._feature_importance,
        )

    def _on_train_complete(self, X_scaled: np.ndarray, y: np.ndarray) -> None:
        """서브클래스에서 추가 학습 후처리 (예: GB 앙상블)."""

    def _predict_proba(self, x_scaled: np.ndarray) -> np.ndarray:
        """예측 확률. 서브클래스에서 앙상블 방식 오버라이드 가능."""
        return self.rf.predict_proba(x_scaled)[0]

    def _build_feature_vector(self, features: dict[str, float]) -> np.ndarray:
        """특성 딕셔너리 → 고정 순서 벡터."""
        return np.array([features.get(f, 0.0) for f in self.feature_names]).reshape(1, -1)

    def _predict_base(self, features: dict[str, float]) -> tuple[np.ndarray, int, int]:
        """공통 예측 로직. Returns: (proba, pred_idx, pred_class)."""
        if not self.is_trained:
            self.train()

        x = self._build_feature_vector(features)
        x_scaled = self.scaler.transform(x)
        proba = self._predict_proba(x_scaled)
        pred_idx = int(np.argmax(proba))
        pred_class = self.classes_[pred_idx]
        return proba, pred_idx, pred_class

    def _build_probabilities(self, proba: np.ndarray) -> dict[str, float]:
        """확률 배열 → {한국어명: 확률} 매핑."""
        probabilities = {}
        for i, cls in enumerate(self.classes_):
            _, label_kr = self.labels[cls]
            probabilities[label_kr] = float(proba[i])
        return probabilities

    def _build_top3(self, proba: np.ndarray) -> list[tuple[str, float]]:
        """확률 배열 → Top 3 리스트."""
        top_indices = np.argsort(proba)[::-1][:3]
        return [
            (self.labels[self.classes_[i]][1], float(proba[i]))
            for i in top_indices
        ]
