"""공통 유틸리티: 심각도 스코어링, 한국어 이름 매핑, 압력 비율 계산.

분석 모듈 전체에서 중복되던 로직을 단일 소스로 통합합니다.
"""

import numpy as np

EPSILON = 1e-8  # 수치 안정성 — 0 나눗셈 방지


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 심각도 스코어링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def severity_label(score: float) -> str:
    """0~1 위험 점수를 한국어 심각도 라벨로 변환.

    Returns: "정상" | "경미" | "주의" | "경고" | "위험"
    """
    if score >= 0.75:
        return "위험"
    elif score >= 0.50:
        return "경고"
    elif score >= 0.25:
        return "주의"
    elif score > 0.0:
        return "경미"
    else:
        return "정상"


def linear_risk_score(
    value: float,
    low_risk: float,
    high_risk: float,
) -> float:
    """두 임계값 사이에서 선형 위험 점수를 계산 (0~1 클램핑).

    low_risk < high_risk: value가 높을수록 위험
    low_risk > high_risk: value가 낮을수록 위험
    """
    if high_risk > low_risk:
        return float(np.clip(
            (value - low_risk) / (high_risk - low_risk + EPSILON), 0, 1
        ))
    else:
        return float(np.clip(
            (low_risk - value) / (low_risk - high_risk + EPSILON), 0, 1
        ))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 한국어 특성 이름 매핑 (Single Source of Truth)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FEATURE_KOREAN = {
    "gait_speed": "보행 속도",
    "cadence": "보행률",
    "cadence_low": "보행률(저)",
    "cadence_high": "보행률(고)",
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
    "anomaly_score": "이상 점수",
    "ml_index": "내외측 체중 분포",
    "ap_index": "전후방 체중 분포",
}


def get_feature_korean(feature_name: str) -> str:
    """특성 이름의 한국어 변환. 매핑에 없으면 원본 반환."""
    return FEATURE_KOREAN.get(feature_name, feature_name)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 압력 비율 파생 특성 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# zone feature 키 이름
_HEEL_ZONES = ["zone_heel_medial_mean", "zone_heel_lateral_mean"]
_FORE_ZONES = ["zone_forefoot_medial_mean", "zone_forefoot_lateral_mean", "zone_toes_mean"]
_MID_ZONES = ["zone_midfoot_medial_mean", "zone_midfoot_lateral_mean"]


def compute_pressure_ratios(features: dict[str, float]) -> dict[str, float]:
    """zone features에서 압력 비율을 계산하여 반환.

    Returns:
        {"heel_pressure_ratio": ..., "forefoot_pressure_ratio": ..., "midfoot_pressure_ratio": ...}
    """
    heel_sum = sum(features.get(z, 0) for z in _HEEL_ZONES)
    fore_sum = sum(features.get(z, 0) for z in _FORE_ZONES)
    mid_sum = sum(features.get(z, 0) for z in _MID_ZONES)
    total = heel_sum + fore_sum + mid_sum + 1e-8

    return {
        "heel_pressure_ratio": heel_sum / total,
        "forefoot_pressure_ratio": fore_sum / total,
        "midfoot_pressure_ratio": mid_sum / total,
    }


def compute_derived_features(features: dict[str, float]) -> None:
    """공통 파생 특성을 in-place로 계산.

    압력 비율, 비대칭 지수, 좌우 변동성, 체간 흔들림, 보행 속도 추정.
    Pydantic model_dump() 결과의 None 값도 올바르게 채운다.
    """
    def _missing(key: str) -> bool:
        return features.get(key) is None

    # 압력 비율
    for key, val in compute_pressure_ratios(features).items():
        if _missing(key):
            features[key] = val

    # 좌우 압력 비대칭
    if not _missing("ml_index") and _missing("pressure_asymmetry"):
        features["pressure_asymmetry"] = abs(features["ml_index"])

    # 좌우 흔들림 변동성
    if not _missing("cop_sway") and _missing("ml_variability"):
        features["ml_variability"] = features["cop_sway"] * 1.5

    # 가속도 변동성
    if not _missing("acceleration_rms") and _missing("acceleration_variability"):
        features["acceleration_variability"] = features["acceleration_rms"] * 0.2

    # 체간 흔들림 (가속도 기반 추정)
    if not _missing("acceleration_rms") and _missing("trunk_sway"):
        features["trunk_sway"] = features["acceleration_rms"] * 1.2

    # 보행 속도 추정 (보행률에서)
    if not _missing("cadence") and _missing("gait_speed"):
        features["gait_speed"] = features["cadence"] / 60.0 * 0.75 / 2.0
