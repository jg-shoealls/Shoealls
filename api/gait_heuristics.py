"""룰 기반 보행 지표·위험도 헬퍼 (진단 아님, 모니터링·리포트용).

프론트엔드 `gaitAnalysis` / `riskAssessment`와 개념을 맞추며,
향후 배치 작업·오프라인 분석에서 재사용할 수 있습니다.
"""

from __future__ import annotations

from typing import TypedDict


class FallRiskHeuristic(TypedDict):
    index_0_100: int
    band: str  # "low" | "moderate" | "high"
    notes: list[str]


def compute_fall_risk_index(
    gait_speed: float,
    stride_regularity: float,
    step_symmetry: float,
    cop_sway: float,
    ml_index: float,
) -> int:
    """특성 기반 낙상 위험 지수 0~100 (높을수록 위험)."""

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    symmetry_risk = clamp01((0.85 - step_symmetry) / 0.35)
    sway_risk = clamp01((cop_sway - 0.04) / 0.12)
    rhythm_risk = clamp01((0.65 - stride_regularity) / 0.5)
    speed_risk = clamp01((1.0 - gait_speed) / 0.9)
    ml_risk = clamp01(ml_index / 0.35)

    raw = (
        symmetry_risk * 0.22
        + sway_risk * 0.20
        + rhythm_risk * 0.18
        + speed_risk * 0.15
        + ml_risk * 0.10
        + (sway_risk + symmetry_risk) * 0.075
    )
    return int(round(clamp01(raw) * 100))


def interpret_fall_risk(index: int) -> FallRiskHeuristic:
    if index < 35:
        band = "low"
    elif index < 65:
        band = "moderate"
    else:
        band = "high"

    notes = [
        "의료 확진이 아닌 센서 기반 위험 징후입니다.",
        "반복 관찰 시 전문 의료기관 방문을 검토하세요.",
    ]
    return {"index_0_100": index, "band": band, "notes": notes}
