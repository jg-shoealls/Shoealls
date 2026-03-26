"""Injury risk engine: assesses 6 types of foot/gait injury risk from pressure data."""

import numpy as np
from dataclasses import dataclass

from .foot_zones import FootZoneAnalyzer, FootAnalysisResult, REGION_GROUPS


@dataclass
class InjuryRisk:
    """Risk assessment for a single injury type."""
    name: str
    korean_name: str
    risk_score: float       # 0-1
    severity: str           # 정상/주의/경고/위험
    contributing_factors: list[str]
    recommendation: str


@dataclass
class InjuryRiskReport:
    """Complete injury risk report."""
    risks: list[InjuryRisk]
    overall_risk: float     # 0-1
    top_risk: str           # name of highest risk
    summary_kr: str         # Korean summary


# Normal reference ranges (from biomechanics literature)
NORMAL_RANGES = {
    "heel_pressure_ratio": (0.25, 0.40),    # heel should bear 25-40% of total
    "forefoot_pressure_ratio": (0.35, 0.55), # forefoot 35-55%
    "midfoot_pressure_ratio": (0.05, 0.20),  # midfoot 5-20%
    "ml_index": (-0.15, 0.15),               # mediolateral balance
    "arch_index": (0.15, 0.35),              # normal arch
    "peak_heel_pressure": (0, 3.0),          # relative scale
    "peak_forefoot_pressure": (0, 2.5),
    "cop_sway": (0, 0.08),
}


class InjuryRiskEngine:
    """Evaluates 6 types of injury risk from foot pressure analysis.

    Injury types:
    1. Plantar fasciitis (족저근막염)
    2. Metatarsal stress fracture (중족골 피로골절)
    3. Ankle sprain risk (발목 염좌 위험)
    4. Heel spur / calcaneal stress (종골 스트레스)
    5. Flat foot / overpronation (평발/과회내)
    6. High arch / supination (요족/과회외)
    """

    def __init__(self, grid_h: int = 16, grid_w: int = 8):
        self.foot_analyzer = FootZoneAnalyzer(grid_h, grid_w)

    def assess_risk(self, pressure_seq: np.ndarray) -> InjuryRiskReport:
        """Run full injury risk assessment on a pressure sequence.

        Args:
            pressure_seq: Shape (T, 1, H, W) or (T, H, W).
        """
        analysis = self.foot_analyzer.analyze_sequence(pressure_seq)
        frames = analysis["frames"]

        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics(frames, analysis)

        # Assess each injury type
        risks = [
            self._assess_plantar_fasciitis(metrics),
            self._assess_metatarsal_stress(metrics),
            self._assess_ankle_sprain(metrics),
            self._assess_heel_spur(metrics),
            self._assess_flat_foot(metrics),
            self._assess_high_arch(metrics),
        ]

        # Overall risk
        risk_scores = [r.risk_score for r in risks]
        overall = float(max(risk_scores))
        top_risk = risks[int(np.argmax(risk_scores))].korean_name

        summary = self._generate_summary(risks, overall)

        return InjuryRiskReport(
            risks=risks,
            overall_risk=overall,
            top_risk=top_risk,
            summary_kr=summary,
        )

    def _compute_aggregate_metrics(self, frames: list[FootAnalysisResult], analysis: dict) -> dict:
        """Compute aggregate metrics across all frames."""
        total_pressures = [f.total_pressure for f in frames]
        avg_total = np.mean(total_pressures) if total_pressures else 1.0

        # Zone pressure ratios
        zone_totals = {}
        for zone_name in ["toes", "forefoot_medial", "forefoot_lateral",
                          "midfoot_medial", "midfoot_lateral",
                          "heel_medial", "heel_lateral"]:
            vals = [f.zone_metrics[zone_name].pressure_integral for f in frames]
            zone_totals[zone_name] = float(np.mean(vals))

        heel_total = zone_totals["heel_medial"] + zone_totals["heel_lateral"]
        fore_total = (zone_totals["toes"] + zone_totals["forefoot_medial"]
                      + zone_totals["forefoot_lateral"])
        mid_total = zone_totals["midfoot_medial"] + zone_totals["midfoot_lateral"]
        grand_total = heel_total + fore_total + mid_total + 1e-8

        # Peak pressures per region
        heel_peaks = [max(f.zone_metrics["heel_medial"].peak_pressure,
                         f.zone_metrics["heel_lateral"].peak_pressure) for f in frames]
        fore_peaks = [max(f.zone_metrics["forefoot_medial"].peak_pressure,
                         f.zone_metrics["forefoot_lateral"].peak_pressure,
                         f.zone_metrics["toes"].peak_pressure) for f in frames]

        return {
            "heel_pressure_ratio": heel_total / grand_total,
            "forefoot_pressure_ratio": fore_total / grand_total,
            "midfoot_pressure_ratio": mid_total / grand_total,
            "ml_index_mean": analysis["ml_index_mean"],
            "ml_index_std": analysis["ml_index_std"],
            "ap_index_mean": analysis["ap_index_mean"],
            "cop_sway": analysis["cop_sway"],
            "arch_index": float(np.mean([f.arch_index for f in frames])),
            "peak_heel_pressure": float(np.max(heel_peaks)) if heel_peaks else 0.0,
            "peak_forefoot_pressure": float(np.max(fore_peaks)) if fore_peaks else 0.0,
            "heel_medial_ratio": zone_totals["heel_medial"] / (heel_total + 1e-8),
            "forefoot_medial_ratio": zone_totals["forefoot_medial"] / (fore_total + 1e-8),
            "midfoot_contact": float(np.mean([
                f.zone_metrics["midfoot_medial"].contact_area_ratio +
                f.zone_metrics["midfoot_lateral"].contact_area_ratio
                for f in frames
            ])) / 2.0,
        }

    def _score(self, value: float, low_risk: float, high_risk: float) -> float:
        """Linear risk scoring between thresholds, clamped to 0-1."""
        if high_risk > low_risk:
            return float(np.clip((value - low_risk) / (high_risk - low_risk + 1e-8), 0, 1))
        else:
            return float(np.clip((low_risk - value) / (low_risk - high_risk + 1e-8), 0, 1))

    def _severity_label(self, score: float) -> str:
        if score < 0.25:
            return "정상"
        elif score < 0.5:
            return "주의"
        elif score < 0.75:
            return "경고"
        else:
            return "위험"

    def _assess_plantar_fasciitis(self, m: dict) -> InjuryRisk:
        """Plantar fasciitis: high heel impact + reduced arch support."""
        factors = []
        scores = []

        # High heel peak pressure
        s1 = self._score(m["peak_heel_pressure"], 2.0, 4.0)
        scores.append(s1)
        if s1 > 0.3:
            factors.append("뒤꿈치 충격 압력 과다")

        # Low arch index (flat)
        s2 = self._score(m["arch_index"], 0.15, 0.05)
        scores.append(s2 * 0.7)
        if s2 > 0.3:
            factors.append("아치 지지 부족")

        # High heel pressure ratio
        s3 = self._score(m["heel_pressure_ratio"], 0.40, 0.60)
        scores.append(s3 * 0.8)
        if s3 > 0.3:
            factors.append("뒤꿈치 하중 비율 과다")

        risk = float(np.clip(np.mean(scores), 0, 1))
        return InjuryRisk(
            name="plantar_fasciitis",
            korean_name="족저근막염",
            risk_score=risk,
            severity=self._severity_label(risk),
            contributing_factors=factors or ["특이사항 없음"],
            recommendation="뒤꿈치 쿠션이 좋은 신발 착용, 아치 지지 인솔 사용, 스트레칭 권장"
            if risk > 0.25 else "현재 양호합니다",
        )

    def _assess_metatarsal_stress(self, m: dict) -> InjuryRisk:
        """Metatarsal stress fracture: excessive forefoot pressure."""
        factors = []
        scores = []

        s1 = self._score(m["forefoot_pressure_ratio"], 0.55, 0.75)
        scores.append(s1)
        if s1 > 0.3:
            factors.append("앞발 하중 비율 과다")

        s2 = self._score(m["peak_forefoot_pressure"], 2.0, 4.0)
        scores.append(s2)
        if s2 > 0.3:
            factors.append("앞발 최고 압력 과다")

        risk = float(np.clip(np.mean(scores), 0, 1))
        return InjuryRisk(
            name="metatarsal_stress",
            korean_name="중족골 피로골절",
            risk_score=risk,
            severity=self._severity_label(risk),
            contributing_factors=factors or ["특이사항 없음"],
            recommendation="앞발 하중을 줄이는 보행 훈련, 메타타르살 패드 사용 권장"
            if risk > 0.25 else "현재 양호합니다",
        )

    def _assess_ankle_sprain(self, m: dict) -> InjuryRisk:
        """Ankle sprain: excessive lateral loading or COP sway."""
        factors = []
        scores = []

        # Lateral shift
        s1 = self._score(abs(m["ml_index_mean"]), 0.15, 0.40)
        scores.append(s1)
        if s1 > 0.3:
            factors.append("좌우 체중 불균형 심함")

        # COP sway
        s2 = self._score(m["cop_sway"], 0.08, 0.20)
        scores.append(s2)
        if s2 > 0.3:
            factors.append("체중심 흔들림 과다")

        # ML variability
        s3 = self._score(m["ml_index_std"], 0.10, 0.25)
        scores.append(s3 * 0.7)
        if s3 > 0.3:
            factors.append("좌우 분포 변동성 높음")

        risk = float(np.clip(np.mean(scores), 0, 1))
        return InjuryRisk(
            name="ankle_sprain",
            korean_name="발목 염좌",
            risk_score=risk,
            severity=self._severity_label(risk),
            contributing_factors=factors or ["특이사항 없음"],
            recommendation="발목 안정성 강화 운동, 불안정한 지면 보행 주의"
            if risk > 0.25 else "현재 양호합니다",
        )

    def _assess_heel_spur(self, m: dict) -> InjuryRisk:
        """Heel spur / calcaneal stress: concentrated heel impact."""
        factors = []
        scores = []

        s1 = self._score(m["peak_heel_pressure"], 2.5, 5.0)
        scores.append(s1)
        if s1 > 0.3:
            factors.append("뒤꿈치 충격 과다")

        s2 = self._score(m["heel_pressure_ratio"], 0.40, 0.55)
        scores.append(s2 * 0.8)
        if s2 > 0.3:
            factors.append("뒤꿈치 체중 집중")

        # Heel medial concentration
        if m["heel_medial_ratio"] > 0.65 or m["heel_medial_ratio"] < 0.35:
            scores.append(0.5)
            factors.append("뒤꿈치 내외측 불균형")
        else:
            scores.append(0.0)

        risk = float(np.clip(np.mean(scores), 0, 1))
        return InjuryRisk(
            name="heel_spur",
            korean_name="종골 스트레스",
            risk_score=risk,
            severity=self._severity_label(risk),
            contributing_factors=factors or ["특이사항 없음"],
            recommendation="뒤꿈치 충격 흡수 패드 사용, 딱딱한 바닥 보행 자제"
            if risk > 0.25 else "현재 양호합니다",
        )

    def _assess_flat_foot(self, m: dict) -> InjuryRisk:
        """Flat foot / overpronation: excessive midfoot contact."""
        factors = []
        scores = []

        # High arch index = high midfoot contact = flat
        s1 = self._score(m["arch_index"], 0.35, 0.60)
        scores.append(s1)
        if s1 > 0.3:
            factors.append("중족부 접촉 면적 과다")

        s2 = self._score(m["midfoot_pressure_ratio"], 0.20, 0.40)
        scores.append(s2)
        if s2 > 0.3:
            factors.append("중족부 하중 과다")

        # Medial shift
        if m["ml_index_mean"] < -0.10:
            scores.append(self._score(-m["ml_index_mean"], 0.10, 0.30))
            factors.append("내측 쏠림 (과회내)")
        else:
            scores.append(0.0)

        risk = float(np.clip(np.mean(scores), 0, 1))
        return InjuryRisk(
            name="flat_foot",
            korean_name="평발/과회내",
            risk_score=risk,
            severity=self._severity_label(risk),
            contributing_factors=factors or ["특이사항 없음"],
            recommendation="아치 지지 인솔 사용, 내측 지지 강화 신발 착용 권장"
            if risk > 0.25 else "현재 양호합니다",
        )

    def _assess_high_arch(self, m: dict) -> InjuryRisk:
        """High arch / supination: minimal midfoot contact."""
        factors = []
        scores = []

        # Very low arch index = no midfoot contact = high arch
        s1 = self._score(m["arch_index"], 0.10, 0.02)
        scores.append(s1)
        if s1 > 0.3:
            factors.append("중족부 접촉 면적 매우 적음")

        s2 = self._score(m["midfoot_pressure_ratio"], 0.05, 0.01)
        scores.append(s2)
        if s2 > 0.3:
            factors.append("중족부 하중 미미")

        # Lateral shift
        if m["ml_index_mean"] > 0.10:
            scores.append(self._score(m["ml_index_mean"], 0.10, 0.30))
            factors.append("외측 쏠림 (과회외)")
        else:
            scores.append(0.0)

        risk = float(np.clip(np.mean(scores), 0, 1))
        return InjuryRisk(
            name="high_arch",
            korean_name="요족/과회외",
            risk_score=risk,
            severity=self._severity_label(risk),
            contributing_factors=factors or ["특이사항 없음"],
            recommendation="쿠션이 좋은 중립 신발 착용, 발바닥 스트레칭 권장"
            if risk > 0.25 else "현재 양호합니다",
        )

    def _generate_summary(self, risks: list[InjuryRisk], overall: float) -> str:
        """Generate Korean summary of injury risk assessment."""
        lines = []
        if overall < 0.25:
            lines.append("전체적으로 발 건강 상태가 양호합니다.")
        elif overall < 0.5:
            lines.append("일부 항목에서 주의가 필요합니다.")
        elif overall < 0.75:
            lines.append("부상 위험 요소가 감지되었습니다. 주의가 필요합니다.")
        else:
            lines.append("높은 부상 위험이 감지되었습니다. 전문가 상담을 권장합니다.")

        warnings = [r for r in risks if r.risk_score >= 0.25]
        if warnings:
            lines.append("\n주요 위험 요소:")
            for r in sorted(warnings, key=lambda x: -x.risk_score):
                lines.append(f"  - {r.korean_name} ({r.severity}): {', '.join(r.contributing_factors)}")

        return "\n".join(lines)
