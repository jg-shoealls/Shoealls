"""종단 질환 변화 감지 엔진 (Longitudinal Disease Change Monitor).

개인별 1년 보행 패턴 기준선을 학습하고, 이로부터의 비정상 변화를 감지하여
파킨슨병·치매·뇌출혈·근골격계 질환의 조기 징후를 포착합니다.

핵심 알고리즘:
  1. AdaptiveBaselineBuilder: 365일 롤링 윈도우 + EWMA 가중 기준선
  2. ChangePointDetector: CUSUM + EWMA + 슬라이딩 윈도우 비교
  3. DiseaseSignatureMatcher: 질환별 다특성 악화 시그니처 매칭
  4. ProgressiveAlertSystem: 관심→주의→경고→위험 4단계 알림

센서 입력:
  - 압력 센서: 13개 바이오마커 (zone pressure, ratios, asymmetry)
  - 속도 센서: gait_speed, cadence, stride_regularity
  - 각속도 센서 (자이로): trunk_sway, ml_variability, acceleration_*
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from .common import get_feature_korean, severity_label


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 질환별 악화 시그니처 정의
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 각 질환의 전형적 보행 악화 패턴:
#   - progression: "gradual" (수개월), "subacute" (수주), "acute" (수일)
#   - feature_changes: 특성별 (변화방향, 최소 z-score, 가중치)
#   - min_features: 최소 몇 개의 특성이 이상 반응해야 질환 후보로 인정되는지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DISEASE_SIGNATURES = {
    "parkinsons": {
        "korean_name": "파킨슨병",
        "progression": "gradual",  # 수개월에 걸친 점진적 악화
        "feature_changes": {
            "stride_regularity":       ("decrease", 1.5, 1.0),  # 보폭 규칙성 저하 (핵심)
            "gait_speed":              ("decrease", 1.5, 0.9),  # 보행 속도 저하
            "cadence":                 ("increase", 1.2, 0.8),  # festination (과속 보행)
            "acceleration_variability": ("increase", 1.5, 0.9),  # 떨림 증가
            "cop_sway":                ("increase", 1.2, 0.6),  # 균형 불안정
            "step_symmetry":           ("decrease", 1.2, 0.5),  # 대칭성 저하
        },
        "min_features": 3,
        "window_days": 60,  # 60일 내 악화 감지
        "referral": "신경과",
        "key_clinical_signs": [
            "소보행 (보폭 단축 + 보행률 증가)",
            "동결 보행 (갑작스러운 멈춤)",
            "움직임 초기 지연",
            "팔 흔들림 감소",
        ],
    },
    "dementia": {
        "korean_name": "치매 (알츠하이머)",
        "progression": "gradual",  # 매우 느린 진행
        "feature_changes": {
            "gait_speed":              ("decrease", 1.2, 1.0),  # 보행 속도 저하 (핵심)
            "cadence":                 ("decrease", 1.0, 0.7),  # 보행률 저하
            "stride_regularity":       ("decrease", 1.2, 0.9),  # 불규칙성
            "step_symmetry":           ("decrease", 1.0, 0.6),
            "trunk_sway":              ("increase", 1.2, 0.7),  # 체간 동요
            "cop_sway":                ("increase", 1.2, 0.6),
        },
        "min_features": 3,
        "window_days": 90,  # 3개월 관찰 권장
        "referral": "신경과 / 치매안심센터",
        "key_clinical_signs": [
            "보행 속도 점진적 저하 (> 0.1 m/s/년)",
            "듀얼태스킹 시 보행 악화",
            "보폭 변동성 증가",
            "균형감각 저하",
        ],
    },
    "cerebral_hemorrhage": {
        "korean_name": "뇌출혈 / 뇌경색",
        "progression": "acute",  # 급성 발병
        "feature_changes": {
            "step_symmetry":           ("decrease", 2.0, 1.0),  # 급격한 비대칭 (핵심)
            "pressure_asymmetry":      ("increase", 2.0, 1.0),  # 좌우 압력 차이
            "gait_speed":              ("decrease", 1.8, 0.9),  # 급격한 속도 저하
            "ml_variability":          ("increase", 1.5, 0.8),  # 좌우 흔들림
            "stride_regularity":       ("decrease", 1.5, 0.7),
        },
        "min_features": 2,  # 급성이므로 소수 특성만으로도 의심
        "window_days": 7,   # 1주일 내 급변 감지
        "referral": "응급실 / 신경과",
        "key_clinical_signs": [
            "급격한 좌우 비대칭 발생",
            "환측 하지 회선(circumduction)",
            "급성 보행 속도 저하",
            "편측 마비 징후",
        ],
    },
    "musculoskeletal": {
        "korean_name": "근골격계 이상 (관절염·디스크·척추관협착)",
        "progression": "subacute",  # 수주~수개월
        "feature_changes": {
            "pressure_asymmetry":      ("increase", 1.5, 0.9),  # 통증 회피 비대칭
            "gait_speed":              ("decrease", 1.2, 0.8),  # 통증으로 인한 감속
            "step_symmetry":           ("decrease", 1.2, 0.7),
            "heel_pressure_ratio":     ("decrease", 1.2, 0.6),  # 뒤꿈치 회피
            "forefoot_pressure_ratio": ("increase", 1.2, 0.6),  # 앞발 집중
            "cadence":                 ("decrease", 1.0, 0.5),
        },
        "min_features": 2,
        "window_days": 30,  # 1개월
        "referral": "정형외과 / 재활의학과",
        "key_clinical_signs": [
            "통증 회피성 절뚝거림",
            "단측 체중 부하 감소",
            "보행 속도 및 보행률 저하",
            "압력 분포 좌우 차이",
        ],
    },
}

# 알림 단계 (4단계)
ALERT_LEVELS = {
    "관심": {"threshold": 0.25, "color": "#2196F3", "icon": "ℹ"},
    "주의": {"threshold": 0.50, "color": "#F39C12", "icon": "⚠"},
    "경고": {"threshold": 0.70, "color": "#E67E22", "icon": "⚠"},
    "위험": {"threshold": 0.85, "color": "#C0392B", "icon": "🚨"},
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 구조
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SessionRecord:
    """단일 보행 세션 기록."""
    timestamp: datetime
    features: dict[str, float]  # 13개 보행 바이오마커

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionRecord":
        return cls(
            timestamp=datetime.fromisoformat(d["timestamp"]),
            features=d["features"],
        )


@dataclass
class PersonalBaseline:
    """개인 1년 보행 기준선."""
    user_id: str
    built_at: datetime
    session_count: int
    observation_days: int
    # feature → (mean, std, ewma_mean, ewma_std, p25, p75)
    stats: dict[str, dict[str, float]] = field(default_factory=dict)
    # 최근 30일 평균 (급성 비교용)
    recent_30d_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def has_feature(self, name: str) -> bool:
        return name in self.stats and self.stats[name].get("std", 0) > 1e-6


@dataclass
class ChangeSignal:
    """단일 특성의 변화 신호."""
    feature: str
    korean_name: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    direction: str          # "increase" | "decrease" | "stable"
    cusum_score: float      # CUSUM 누적 일탈 점수
    ewma_deviation: float   # EWMA 기반 평활 편차
    sliding_window_shift: float  # 최근 14일 vs 90일 평균 차이
    aggregate_score: float  # 종합 변화 점수 0~1


@dataclass
class DiseaseAlert:
    """질환 악화 경고."""
    disease_id: str
    korean_name: str
    alert_level: str         # 관심 | 주의 | 경고 | 위험
    alert_score: float       # 0~1
    matched_signals: list[ChangeSignal]
    progression_type: str
    days_observed: int
    referral: str
    key_signs: list[str]
    recommendation_kr: str


@dataclass
class LongitudinalReport:
    """1년 종단 분석 종합 보고서."""
    user_id: str
    report_date: datetime
    baseline: PersonalBaseline
    change_signals: list[ChangeSignal]  # 이상 반응한 특성
    disease_alerts: list[DiseaseAlert]  # 감지된 질환 경고
    overall_health_trend: str           # "양호" | "주의" | "악화"
    summary_kr: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AdaptiveBaselineBuilder: 1년 기준선 구축
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdaptiveBaselineBuilder:
    """1년치 보행 세션에서 개인 기준선을 구축합니다.

    알고리즘:
      1. 최근 365일 세션을 롤링 윈도우로 수집
      2. 각 특성별 평균/표준편차 + EWMA(지수가중평균) 계산
      3. 외곽값(3σ 이상) 1회 제거 후 재계산 (robust baseline)
      4. 최근 30일 별도 통계 (급성 비교용)

    EWMA 가중치는 최근 데이터에 더 높은 비중을 부여합니다.
    """

    DEFAULT_EWMA_ALPHA = 0.1      # 낮을수록 장기 평균
    BASELINE_DAYS = 365            # 1년
    RECENT_COMPARISON_DAYS = 30    # 최근 비교용 윈도우
    MIN_SESSIONS = 20              # 최소 세션 수
    OUTLIER_Z_THRESHOLD = 3.0      # 외곽값 기준

    def __init__(self, ewma_alpha: float = DEFAULT_EWMA_ALPHA):
        self.ewma_alpha = ewma_alpha

    def build(
        self,
        sessions: list[SessionRecord],
        user_id: str,
        reference_date: Optional[datetime] = None,
    ) -> PersonalBaseline:
        """세션 기록에서 개인 기준선을 구축합니다."""
        if reference_date is None:
            reference_date = datetime.now()

        cutoff_365 = reference_date - timedelta(days=self.BASELINE_DAYS)
        cutoff_30 = reference_date - timedelta(days=self.RECENT_COMPARISON_DAYS)

        # 1년 이내 세션 필터
        year_sessions = [s for s in sessions if s.timestamp >= cutoff_365]
        recent_sessions = [s for s in sessions if s.timestamp >= cutoff_30]

        if len(year_sessions) < self.MIN_SESSIONS:
            # 세션 부족 - 빈 기준선 반환
            return PersonalBaseline(
                user_id=user_id,
                built_at=reference_date,
                session_count=len(year_sessions),
                observation_days=0,
                stats={},
                recent_30d_stats={},
            )

        # 관찰 기간
        timestamps = [s.timestamp for s in year_sessions]
        observation_days = (max(timestamps) - min(timestamps)).days

        # 특성별 통계 계산
        stats = self._compute_feature_stats(year_sessions, robust=True)
        recent_stats = self._compute_feature_stats(recent_sessions, robust=False)

        return PersonalBaseline(
            user_id=user_id,
            built_at=reference_date,
            session_count=len(year_sessions),
            observation_days=observation_days,
            stats=stats,
            recent_30d_stats=recent_stats,
        )

    def _compute_feature_stats(
        self,
        sessions: list[SessionRecord],
        robust: bool = True,
    ) -> dict[str, dict[str, float]]:
        """특성별 평균/표준편차/EWMA/분위수 계산."""
        if not sessions:
            return {}

        # 시간순 정렬
        sorted_sess = sorted(sessions, key=lambda s: s.timestamp)

        # 모든 특성 수집
        all_features = set()
        for s in sorted_sess:
            all_features.update(s.features.keys())

        stats = {}
        for feat in all_features:
            values = [s.features.get(feat) for s in sorted_sess]
            values = [v for v in values if v is not None and not np.isnan(v)]
            if len(values) < 3:
                continue

            arr = np.array(values, dtype=float)

            # Robust: 외곽값 1회 제거
            if robust and len(arr) >= 5:
                m0, s0 = arr.mean(), arr.std() + 1e-8
                z_scores = np.abs((arr - m0) / s0)
                arr = arr[z_scores <= self.OUTLIER_Z_THRESHOLD]
                if len(arr) < 3:
                    arr = np.array(values)

            # EWMA 계산 (최근 데이터에 가중치)
            ewma_mean, ewma_var = self._compute_ewma(arr, self.ewma_alpha)

            stats[feat] = {
                "mean": float(arr.mean()),
                "std": float(arr.std() + 1e-8),
                "ewma_mean": float(ewma_mean),
                "ewma_std": float(np.sqrt(ewma_var) + 1e-8),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "n": int(len(arr)),
            }
        return stats

    @staticmethod
    def _compute_ewma(values: np.ndarray, alpha: float) -> tuple[float, float]:
        """지수가중 이동평균 및 분산."""
        if len(values) == 0:
            return 0.0, 0.0
        ewma = values[0]
        ewvar = 0.0
        for v in values[1:]:
            diff = v - ewma
            incr = alpha * diff
            ewma = ewma + incr
            ewvar = (1 - alpha) * (ewvar + diff * incr)
        return float(ewma), float(ewvar)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ChangePointDetector: 변화점 감지 알고리즘
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChangePointDetector:
    """기준선 대비 최근 세션의 변화를 감지합니다.

    3가지 통계적 감지 방법을 결합:
      1. Z-score: 단일 세션의 기준선 대비 편차
      2. CUSUM: 연속된 소규모 편차의 누적 (점진적 변화)
      3. Sliding Window 비교: 최근 14일 vs 90일 평균 차이
      4. EWMA 편차: 지수가중 평균 대비 현재 값 편차

    여러 지표를 결합하여 False Positive를 줄입니다.
    """

    CUSUM_K = 0.5          # CUSUM 민감도 (작을수록 민감)
    CUSUM_H = 4.0          # CUSUM 임계값
    RECENT_WINDOW_DAYS = 14
    COMPARISON_WINDOW_DAYS = 90

    def detect(
        self,
        sessions: list[SessionRecord],
        baseline: PersonalBaseline,
        reference_date: Optional[datetime] = None,
    ) -> list[ChangeSignal]:
        """최근 세션 데이터에서 변화 신호를 감지합니다."""
        if not baseline.stats or reference_date is None:
            reference_date = reference_date or datetime.now()

        if not baseline.stats:
            return []

        recent_cutoff = reference_date - timedelta(days=self.RECENT_WINDOW_DAYS)
        compare_cutoff = reference_date - timedelta(days=self.COMPARISON_WINDOW_DAYS)

        recent_sessions = sorted(
            [s for s in sessions if s.timestamp >= recent_cutoff],
            key=lambda s: s.timestamp,
        )
        compare_sessions = [
            s for s in sessions
            if compare_cutoff <= s.timestamp < recent_cutoff
        ]

        if len(recent_sessions) < 2:
            return []

        signals = []
        for feat, feat_stats in baseline.stats.items():
            if feat_stats["std"] < 1e-6:
                continue

            recent_values = [
                s.features.get(feat) for s in recent_sessions
                if s.features.get(feat) is not None
            ]
            compare_values = [
                s.features.get(feat) for s in compare_sessions
                if s.features.get(feat) is not None
            ]

            if len(recent_values) < 2:
                continue

            signal = self._analyze_feature(
                feat, recent_values, compare_values, feat_stats
            )
            if signal is not None:
                signals.append(signal)

        # 종합 변화 점수 순 정렬
        signals.sort(key=lambda s: -s.aggregate_score)
        return signals

    def _analyze_feature(
        self,
        feature: str,
        recent_values: list[float],
        compare_values: list[float],
        baseline_stats: dict[str, float],
    ) -> Optional[ChangeSignal]:
        """단일 특성의 변화 신호 분석."""
        mean = baseline_stats["mean"]
        std = baseline_stats["std"]

        recent_arr = np.array(recent_values, dtype=float)
        current_val = float(recent_arr.mean())

        # 1. Z-score (현재 평균 vs 기준선)
        z_score = (current_val - mean) / (std + 1e-8)
        direction = "increase" if z_score > 0 else ("decrease" if z_score < 0 else "stable")

        # 2. CUSUM: 연속된 편차의 누적 점수
        cusum_pos, cusum_neg = 0.0, 0.0
        max_cusum = 0.0
        for v in recent_arr:
            normalized = (v - mean) / std
            cusum_pos = max(0, cusum_pos + normalized - self.CUSUM_K)
            cusum_neg = max(0, cusum_neg - normalized - self.CUSUM_K)
            max_cusum = max(max_cusum, cusum_pos, cusum_neg)

        # 3. Sliding Window 비교: 최근 14일 vs 이전 90일
        if compare_values:
            compare_arr = np.array(compare_values, dtype=float)
            sliding_shift = (recent_arr.mean() - compare_arr.mean()) / (std + 1e-8)
        else:
            sliding_shift = z_score

        # 4. EWMA 편차: 현재 값이 EWMA 평균에서 얼마나 멀어졌는지
        ewma_mean = baseline_stats.get("ewma_mean", mean)
        ewma_std = baseline_stats.get("ewma_std", std)
        ewma_deviation = (current_val - ewma_mean) / (ewma_std + 1e-8)

        # 종합 변화 점수 (0~1): 여러 지표의 가중 평균
        abs_z = abs(z_score)
        abs_sliding = abs(sliding_shift)
        cusum_norm = min(max_cusum / self.CUSUM_H, 1.0)
        abs_ewma = abs(ewma_deviation)

        aggregate = (
            0.35 * min(abs_z / 3.0, 1.0) +
            0.25 * min(abs_sliding / 2.5, 1.0) +
            0.25 * cusum_norm +
            0.15 * min(abs_ewma / 3.0, 1.0)
        )

        # 유의미한 변화만 반환 (z-score 1.0 이상 or CUSUM 절반 이상)
        if abs_z < 1.0 and cusum_norm < 0.3 and abs_sliding < 1.0:
            return None

        return ChangeSignal(
            feature=feature,
            korean_name=get_feature_korean(feature),
            current_value=current_val,
            baseline_mean=mean,
            baseline_std=std,
            z_score=float(z_score),
            direction=direction,
            cusum_score=float(cusum_norm),
            ewma_deviation=float(ewma_deviation),
            sliding_window_shift=float(sliding_shift),
            aggregate_score=float(np.clip(aggregate, 0, 1)),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DiseaseSignatureMatcher: 질환별 시그니처 매칭
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DiseaseSignatureMatcher:
    """감지된 변화 신호를 질환별 시그니처와 매칭합니다.

    각 질환은 특정 특성들이 특정 방향으로 변화하는 패턴을 가집니다:
      - 파킨슨: stride_regularity↓ + gait_speed↓ + cadence↑ + variability↑
      - 치매: gait_speed↓ + cadence↓ + stride_regularity↓ (느린 진행)
      - 뇌출혈: step_symmetry↓ + pressure_asymmetry↑ (급성)
      - 근골격계: pressure_asymmetry↑ + heel_pressure↓ + forefoot_pressure↑

    매칭 점수는 (일치 특성 비율) × (평균 z-score 강도) × (시그니처 가중치)로 계산.
    """

    def __init__(self, signatures: dict = None):
        self.signatures = signatures or DISEASE_SIGNATURES

    def match(
        self,
        change_signals: list[ChangeSignal],
        observation_days: int,
    ) -> list[DiseaseAlert]:
        """변화 신호를 질환 시그니처와 매칭하여 경고 생성."""
        signals_by_feat = {s.feature: s for s in change_signals}
        alerts = []

        for disease_id, sig in self.signatures.items():
            alert = self._evaluate_signature(
                disease_id, sig, signals_by_feat, observation_days
            )
            if alert is not None:
                alerts.append(alert)

        # 경고 점수 순 정렬
        alerts.sort(key=lambda a: -a.alert_score)
        return alerts

    def _evaluate_signature(
        self,
        disease_id: str,
        signature: dict,
        signals_by_feat: dict[str, ChangeSignal],
        observation_days: int,
    ) -> Optional[DiseaseAlert]:
        """단일 질환 시그니처 매칭."""
        matched_signals = []
        weighted_scores = []
        total_weight = 0.0

        for feat, (expected_dir, min_z, weight) in signature["feature_changes"].items():
            signal = signals_by_feat.get(feat)
            total_weight += weight

            if signal is None:
                continue

            # 방향이 일치하고 최소 z-score를 초과해야 매칭
            if signal.direction == expected_dir and abs(signal.z_score) >= min_z:
                matched_signals.append(signal)
                # 강도 점수: |z-score| / 3.0, 최대 1.0
                intensity = min(abs(signal.z_score) / 3.0, 1.0)
                weighted_scores.append(intensity * weight)

        # 최소 매칭 특성 수 체크
        min_features = signature.get("min_features", 2)
        if len(matched_signals) < min_features:
            return None

        # 경고 점수: (일치 특성 비율) × (평균 강도)
        match_ratio = len(matched_signals) / len(signature["feature_changes"])
        if total_weight > 0:
            avg_intensity = sum(weighted_scores) / total_weight
        else:
            avg_intensity = 0.0

        # 급성 질환은 점수 부스트 (빠른 변화가 더 위험)
        progression = signature.get("progression", "gradual")
        if progression == "acute":
            boost = 1.3
        elif progression == "subacute":
            boost = 1.1
        else:
            boost = 1.0

        alert_score = float(np.clip(match_ratio * avg_intensity * boost, 0, 1))

        if alert_score < 0.2:
            return None

        alert_level = self._score_to_level(alert_score)
        recommendation = self._build_recommendation(
            disease_id, signature, alert_level, matched_signals
        )

        return DiseaseAlert(
            disease_id=disease_id,
            korean_name=signature["korean_name"],
            alert_level=alert_level,
            alert_score=alert_score,
            matched_signals=matched_signals,
            progression_type=progression,
            days_observed=observation_days,
            referral=signature["referral"],
            key_signs=signature["key_clinical_signs"],
            recommendation_kr=recommendation,
        )

    @staticmethod
    def _score_to_level(score: float) -> str:
        """점수 → 알림 단계 변환."""
        if score >= ALERT_LEVELS["위험"]["threshold"]:
            return "위험"
        elif score >= ALERT_LEVELS["경고"]["threshold"]:
            return "경고"
        elif score >= ALERT_LEVELS["주의"]["threshold"]:
            return "주의"
        else:
            return "관심"

    @staticmethod
    def _build_recommendation(
        disease_id: str,
        signature: dict,
        alert_level: str,
        signals: list[ChangeSignal],
    ) -> str:
        """한국어 권고문 생성."""
        referral = signature["referral"]
        progression = signature.get("progression", "gradual")

        if alert_level == "위험":
            if progression == "acute":
                return f"즉시 응급실 방문 권고. 급성 보행 변화가 감지되었습니다. ({referral})"
            return f"가능한 빠른 시일 내 {referral} 진료가 필요합니다. 전문의 상담 권장."
        elif alert_level == "경고":
            return f"{referral} 정밀 검사 권장. 최근 보행 패턴의 유의미한 변화가 지속되고 있습니다."
        elif alert_level == "주의":
            return (
                f"{referral} 방문 고려. 보행 패턴에 변화가 관찰되므로 "
                f"2~4주 후 재측정하여 추세를 확인하시기 바랍니다."
            )
        else:
            return (
                "현재로서는 관찰 단계입니다. 매주 정기적인 보행 측정을 유지하고 "
                "생활 습관(운동·수면·영양)을 점검하세요."
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LongitudinalDiseaseMonitor: 메인 오케스트레이터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LongitudinalDiseaseMonitor:
    """1년 개인 기준선 기반 질환 변화 감지 메인 엔진.

    사용 흐름:
      monitor = LongitudinalDiseaseMonitor()
      monitor.add_session(timestamp, features)  # 세션 누적
      ...
      report = monitor.analyze(user_id="user123")
    """

    def __init__(
        self,
        ewma_alpha: float = 0.1,
        storage_path: Optional[Path] = None,
    ):
        self.baseline_builder = AdaptiveBaselineBuilder(ewma_alpha=ewma_alpha)
        self.change_detector = ChangePointDetector()
        self.signature_matcher = DiseaseSignatureMatcher()
        self.sessions: list[SessionRecord] = []
        self.storage_path = storage_path

    # ── 세션 관리 ────────────────────────────────────────────────────
    def add_session(
        self,
        features: dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> SessionRecord:
        """새 보행 세션 추가."""
        if timestamp is None:
            timestamp = datetime.now()
        record = SessionRecord(timestamp=timestamp, features=dict(features))
        self.sessions.append(record)
        return record

    def load_sessions(self, path: Path):
        """JSON 파일에서 세션 기록 로드."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        self.sessions = [SessionRecord.from_dict(d) for d in data]

    def save_sessions(self, path: Path):
        """세션 기록을 JSON으로 저장."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([s.to_dict() for s in self.sessions], f, indent=2)

    # ── 분석 ─────────────────────────────────────────────────────────
    def analyze(
        self,
        user_id: str = "user",
        reference_date: Optional[datetime] = None,
    ) -> LongitudinalReport:
        """누적된 세션을 분석하여 종단 보고서 생성."""
        if reference_date is None:
            reference_date = datetime.now()

        # 1. 1년 기준선 구축
        baseline = self.baseline_builder.build(
            self.sessions, user_id, reference_date
        )

        if baseline.session_count < AdaptiveBaselineBuilder.MIN_SESSIONS:
            return LongitudinalReport(
                user_id=user_id,
                report_date=reference_date,
                baseline=baseline,
                change_signals=[],
                disease_alerts=[],
                overall_health_trend="양호",
                summary_kr=self._insufficient_data_summary(baseline),
            )

        # 2. 변화점 감지
        signals = self.change_detector.detect(
            self.sessions, baseline, reference_date
        )

        # 3. 질환 시그니처 매칭
        alerts = self.signature_matcher.match(signals, baseline.observation_days)

        # 4. 종합 건강 추세 판단
        trend = self._assess_overall_trend(alerts, signals)

        # 5. 한국어 보고서 생성
        summary = self._build_summary(baseline, signals, alerts, trend)

        return LongitudinalReport(
            user_id=user_id,
            report_date=reference_date,
            baseline=baseline,
            change_signals=signals,
            disease_alerts=alerts,
            overall_health_trend=trend,
            summary_kr=summary,
        )

    def _assess_overall_trend(
        self,
        alerts: list[DiseaseAlert],
        signals: list[ChangeSignal],
    ) -> str:
        """종합 건강 추세 판단."""
        if any(a.alert_level in ("위험", "경고") for a in alerts):
            return "악화"
        if any(a.alert_level == "주의" for a in alerts):
            return "주의"
        if signals and max(s.aggregate_score for s in signals) > 0.5:
            return "주의"
        return "양호"

    # ── 보고서 ───────────────────────────────────────────────────────
    @staticmethod
    def _insufficient_data_summary(baseline: PersonalBaseline) -> str:
        needed = AdaptiveBaselineBuilder.MIN_SESSIONS - baseline.session_count
        return (
            f"[데이터 부족] 현재 {baseline.session_count}회 세션 기록.\n"
            f"정확한 기준선 구축을 위해 추가 {needed}회 이상의 측정이 필요합니다.\n"
            f"주 3~4회, 최소 4주간 꾸준히 측정하시기 바랍니다."
        )

    def _build_summary(
        self,
        baseline: PersonalBaseline,
        signals: list[ChangeSignal],
        alerts: list[DiseaseAlert],
        trend: str,
    ) -> str:
        """종합 한국어 보고서."""
        lines = []
        lines.append("=" * 64)
        lines.append("  개인 보행 종단 분석 보고서 (1년 기준선 기반)")
        lines.append("=" * 64)
        lines.append(f"  사용자: {baseline.user_id}")
        lines.append(f"  분석 일자: {baseline.built_at.strftime('%Y-%m-%d')}")
        lines.append(f"  관찰 기간: {baseline.observation_days}일 "
                     f"(세션 {baseline.session_count}회)")
        lines.append(f"  종합 건강 추세: {trend}")
        lines.append("")

        # 기준선 요약
        lines.append("-" * 64)
        lines.append("  [개인 기준선 요약]")
        lines.append("-" * 64)
        key_features = ["gait_speed", "cadence", "stride_regularity",
                        "step_symmetry", "cop_sway", "trunk_sway"]
        for feat in key_features:
            if feat in baseline.stats:
                s = baseline.stats[feat]
                kname = get_feature_korean(feat)
                lines.append(
                    f"    {kname:<12}: 평균 {s['mean']:.3f} ± {s['std']:.3f} "
                    f"(IQR {s['p25']:.3f}~{s['p75']:.3f})"
                )
        lines.append("")

        # 변화 신호
        if signals:
            lines.append("-" * 64)
            lines.append(f"  [감지된 변화 신호: {len(signals)}개 특성]")
            lines.append("-" * 64)
            for sig in signals[:8]:
                arrow = "↑" if sig.direction == "increase" else "↓"
                lines.append(
                    f"    {sig.korean_name} {arrow} "
                    f"(z={sig.z_score:+.2f}, CUSUM={sig.cusum_score:.2f}, "
                    f"변화점수={sig.aggregate_score:.2f})"
                )
            lines.append("")

        # 질환 경고
        if alerts:
            lines.append("-" * 64)
            lines.append(f"  [질환 변화 경고: {len(alerts)}건]")
            lines.append("-" * 64)
            for alert in alerts:
                icon = ALERT_LEVELS[alert.alert_level]["icon"]
                lines.append(
                    f"  {icon} [{alert.alert_level}] {alert.korean_name} "
                    f"(경고점수: {alert.alert_score:.0%})"
                )
                lines.append(f"     진행 유형: {alert.progression_type}")
                lines.append(f"     매칭 특성: "
                             f"{', '.join(s.korean_name for s in alert.matched_signals)}")
                lines.append(f"     권고: {alert.recommendation_kr}")
                if alert.key_signs:
                    lines.append("     임상 핵심 징후:")
                    for s in alert.key_signs[:3]:
                        lines.append(f"       · {s}")
                lines.append("")
        else:
            lines.append("  [질환 경고 없음]")
            lines.append("  현재 기준선 대비 유의미한 질환 시그니처가 감지되지 않았습니다.")
            lines.append("")

        lines.append("=" * 64)
        lines.append(f"  ※ 본 분석은 참고용이며, 의학적 진단은 전문의 진료가 필요합니다.")
        lines.append("=" * 64)
        return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 편의 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze_longitudinal(
    sessions: list[dict],
    user_id: str = "user",
    reference_date: Optional[datetime] = None,
) -> LongitudinalReport:
    """세션 리스트에서 종단 분석 실행 (편의 함수).

    Args:
        sessions: [{"timestamp": "...", "features": {...}}, ...]
        user_id: 사용자 ID.
        reference_date: 분석 기준일.
    """
    monitor = LongitudinalDiseaseMonitor()
    for s in sessions:
        ts = s["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        monitor.add_session(s["features"], timestamp=ts)
    return monitor.analyze(user_id, reference_date)
