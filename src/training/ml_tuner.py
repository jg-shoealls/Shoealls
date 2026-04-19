"""ML 분류기 하이퍼파라미터 자동 튜닝.

Optuna를 사용하여 RandomForest + GradientBoosting 앙상블의
최적 하이퍼파라미터를 탐색합니다.

사용법:
    tuner = MLClassifierTuner()
    result = tuner.run(n_trials=100)
    print(result.best_params)
    print(result.best_metrics)

    # 최적 파라미터로 분류기 생성
    clf = result.build_classifier()
    prediction = clf.predict(features)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.analysis.disease_classifier import (
    GaitDiseaseClassifier,
    DISEASE_LABELS,
    FEATURE_NAMES,
    _DISEASE_PROFILES,
)
from src.analysis.config import CLASSIFIER_DEFAULTS

try:
    import optuna
    from optuna.trial import Trial

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@dataclass
class TuningResult:
    """하이퍼파라미터 튜닝 결과."""

    best_params: dict
    best_f1: float
    best_accuracy: float
    cv_scores: list[float]
    all_trials: list[dict] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)

    def build_classifier(self) -> GaitDiseaseClassifier:
        """최적 파라미터로 분류기를 생성하고 학습합니다."""
        p = self.best_params
        clf = GaitDiseaseClassifier(
            n_estimators=p["rf_n_estimators"],
            random_state=CLASSIFIER_DEFAULTS["random_state"],
        )
        clf.rf.set_params(
            max_depth=p["rf_max_depth"],
            min_samples_leaf=p["rf_min_samples_leaf"],
            max_features=p.get("rf_max_features", "sqrt"),
        )
        clf.gb.set_params(
            n_estimators=p["gb_n_estimators"],
            max_depth=p["gb_max_depth"],
            learning_rate=p["gb_learning_rate"],
            subsample=p.get("gb_subsample", 0.8),
        )
        X, y = clf.generate_training_data(
            n_per_class=p.get("samples_per_class", CLASSIFIER_DEFAULTS["samples_per_class"]),
        )
        clf.train(X, y, cv_folds=CLASSIFIER_DEFAULTS["cv_folds"])
        return clf


class MLClassifierTuner:
    """ML 분류기 하이퍼파라미터 Optuna 튜너.

    탐색 공간:
        RandomForest: n_estimators, max_depth, min_samples_leaf, max_features
        GradientBoosting: n_estimators, max_depth, learning_rate, subsample
        데이터: samples_per_class
    """

    def __init__(
        self,
        cv_folds: int = CLASSIFIER_DEFAULTS["cv_folds"],
        random_state: int = CLASSIFIER_DEFAULTS["random_state"],
    ):
        if not HAS_OPTUNA:
            raise ImportError("optuna가 필요합니다: pip install optuna")
        self.cv_folds = cv_folds
        self.random_state = random_state

    def _objective(self, trial: Trial) -> float:
        # RF 파라미터
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 500, step=50)
        rf_max_depth = trial.suggest_int("rf_max_depth", 4, 20)
        rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 2, 20)
        rf_max_features = trial.suggest_categorical(
            "rf_max_features", ["sqrt", "log2", 0.5, 0.7, 0.9]
        )

        # GB 파라미터
        gb_n_estimators = trial.suggest_int("gb_n_estimators", 50, 500, step=50)
        gb_max_depth = trial.suggest_int("gb_max_depth", 3, 10)
        gb_learning_rate = trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True)
        gb_subsample = trial.suggest_float("gb_subsample", 0.6, 1.0)

        # 데이터 규모
        samples_per_class = trial.suggest_categorical(
            "samples_per_class", [50, 100, 200, 300]
        )

        # 앙상블 가중치
        ensemble_weight = trial.suggest_float("ensemble_rf_weight", 0.3, 0.7)

        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            max_features=rf_max_features,
            random_state=self.random_state,
            class_weight="balanced",
        )
        gb = GradientBoostingClassifier(
            n_estimators=gb_n_estimators,
            max_depth=gb_max_depth,
            learning_rate=gb_learning_rate,
            subsample=gb_subsample,
            random_state=self.random_state,
        )
        scaler = StandardScaler()

        X, y = self._generate_data(samples_per_class)
        X_scaled = scaler.fit_transform(X)

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            rf.fit(X_tr, y_tr)
            gb.fit(X_tr, y_tr)

            rf_proba = rf.predict_proba(X_val)
            gb_proba = gb.predict_proba(X_val)
            ensemble_proba = ensemble_weight * rf_proba + (1 - ensemble_weight) * gb_proba
            preds = ensemble_proba.argmax(axis=1)

            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            f1_scores.append(f1)

            trial.report(np.mean(f1_scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(f1_scores))

    def _generate_data(self, n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.random_state)
        X_all, y_all = [], []
        for label, profile in _DISEASE_PROFILES.items():
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

    def run(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> TuningResult:
        """튜닝 실행.

        Args:
            n_trials: 탐색 횟수.
            timeout: 최대 소요 시간(초).
            output_dir: 결과 저장 경로 (None이면 저장 안함).

        Returns:
            TuningResult with best params and metrics.
        """
        study = optuna.create_study(
            direction="maximize",
            study_name="ml_classifier_tuning",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        )
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        best = study.best_params
        best_value = study.best_value

        # 최적 파라미터로 최종 학습하여 상세 메트릭 수집
        final_metrics = self._final_evaluation(best)

        all_trials = [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        result = TuningResult(
            best_params=best,
            best_f1=best_value,
            best_accuracy=final_metrics["accuracy"],
            cv_scores=final_metrics["cv_scores"],
            all_trials=all_trials,
            feature_importance=final_metrics["feature_importance"],
        )

        if output_dir:
            self._save_results(result, study, Path(output_dir))

        return result

    def _final_evaluation(self, params: dict) -> dict:
        """최적 파라미터로 최종 평가."""
        rf = RandomForestClassifier(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_leaf=params["rf_min_samples_leaf"],
            max_features=params.get("rf_max_features", "sqrt"),
            random_state=self.random_state,
            class_weight="balanced",
        )
        gb = GradientBoostingClassifier(
            n_estimators=params["gb_n_estimators"],
            max_depth=params["gb_max_depth"],
            learning_rate=params["gb_learning_rate"],
            subsample=params.get("gb_subsample", 0.8),
            random_state=self.random_state,
        )
        scaler = StandardScaler()
        ensemble_w = params.get("ensemble_rf_weight", 0.5)

        X, y = self._generate_data(
            params.get("samples_per_class", CLASSIFIER_DEFAULTS["samples_per_class"])
        )
        X_scaled = scaler.fit_transform(X)

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        cv_scores = []
        for train_idx, val_idx in cv.split(X_scaled, y):
            rf.fit(X_scaled[train_idx], y[train_idx])
            gb.fit(X_scaled[train_idx], y[train_idx])
            proba = (
                ensemble_w * rf.predict_proba(X_scaled[val_idx])
                + (1 - ensemble_w) * gb.predict_proba(X_scaled[val_idx])
            )
            preds = proba.argmax(axis=1)
            cv_scores.append(accuracy_score(y[val_idx], preds))

        rf.fit(X_scaled, y)
        gb.fit(X_scaled, y)
        avg_imp = (rf.feature_importances_ + gb.feature_importances_) / 2
        importance = {
            FEATURE_NAMES[i]: float(avg_imp[i])
            for i in range(len(FEATURE_NAMES))
        }

        proba = ensemble_w * rf.predict_proba(X_scaled) + (1 - ensemble_w) * gb.predict_proba(X_scaled)
        preds = proba.argmax(axis=1)

        return {
            "accuracy": accuracy_score(y, preds),
            "cv_scores": cv_scores,
            "feature_importance": importance,
        }

    def _save_results(self, result: TuningResult, study, output_dir: Path):
        """결과를 파일로 저장."""
        import json

        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "best_params.json", "w") as f:
            json.dump(
                {
                    "best_params": result.best_params,
                    "best_f1": result.best_f1,
                    "best_accuracy": result.best_accuracy,
                    "cv_scores": result.cv_scores,
                    "feature_importance": result.feature_importance,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        try:
            self._plot_results(study, result, output_dir)
        except Exception:
            pass

    def _plot_results(self, study, result: TuningResult, output_dir: Path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not trials:
            return

        values = [t.value for t in trials]
        best_so_far = np.maximum.accumulate(values)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].scatter(range(len(values)), values, alpha=0.4, s=15, label="Trial F1")
        axes[0].plot(best_so_far, color="red", linewidth=2, label="Best F1")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("F1 Macro")
        axes[0].set_title("Optimization History")
        axes[0].legend()

        sorted_imp = sorted(result.feature_importance.items(), key=lambda x: -x[1])
        names, vals = zip(*sorted_imp)
        axes[1].barh(list(reversed(names)), list(reversed(vals)), color="#2196F3")
        axes[1].set_xlabel("Importance")
        axes[1].set_title("Feature Importance (Best Model)")

        axes[2].bar(range(len(result.cv_scores)), result.cv_scores, color="#4CAF50")
        axes[2].axhline(
            np.mean(result.cv_scores), color="red", linestyle="--", label="Mean"
        )
        axes[2].set_xlabel("Fold")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title("Cross-Validation Scores")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(output_dir / "ml_tuning_results.png", dpi=150, bbox_inches="tight")
        plt.close()
