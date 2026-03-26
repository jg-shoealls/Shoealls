"""Longitudinal trend tracker: session-over-session gait analysis."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrendPoint:
    """Single session data point for trend tracking."""
    session_id: int
    features: dict[str, float]
    injury_risk: float
    overall_deviation: float


@dataclass
class TrendAnalysis:
    """Results of longitudinal trend analysis."""
    metric_trends: dict[str, dict]   # metric -> {direction, slope, p_value_approx, summary_kr}
    improving_metrics: list[str]
    worsening_metrics: list[str]
    stable_metrics: list[str]
    sessions_analyzed: int
    report_kr: str


class LongitudinalTrendTracker:
    """Tracks gait metrics across sessions and identifies trends.

    Uses simple linear regression on metric time series to detect
    improvement, worsening, or stability.
    """

    TREND_THRESHOLD = 0.05  # minimum slope magnitude for significant trend

    METRIC_LABELS = {
        "ml_index": ("내외측 체중 분포", "lower_better"),
        "ap_index": ("전후방 체중 분포", "neutral"),
        "arch_index": ("아치 지수", "neutral"),
        "cop_sway": ("체중심 흔들림", "lower_better"),
        "cadence": ("보행 속도", "neutral"),
        "stride_regularity": ("보폭 규칙성", "higher_better"),
        "step_symmetry": ("좌우 대칭성", "higher_better"),
        "acceleration_rms": ("가속도 크기", "neutral"),
        "injury_risk": ("부상 위험도", "lower_better"),
        "overall_deviation": ("개인 기준 편차", "lower_better"),
    }

    def __init__(self):
        self.history: list[TrendPoint] = []

    def add_session(
        self,
        features: dict[str, float],
        injury_risk: float = 0.0,
        overall_deviation: float = 0.0,
    ):
        """Record a new session's data."""
        sid = len(self.history)
        self.history.append(TrendPoint(
            session_id=sid,
            features=features,
            injury_risk=injury_risk,
            overall_deviation=overall_deviation,
        ))

    def analyze_trends(self, min_sessions: int = 3) -> TrendAnalysis:
        """Analyze trends across recorded sessions.

        Args:
            min_sessions: Minimum sessions required for trend analysis.
        """
        n = len(self.history)
        if n < min_sessions:
            return TrendAnalysis(
                metric_trends={},
                improving_metrics=[],
                worsening_metrics=[],
                stable_metrics=[],
                sessions_analyzed=n,
                report_kr=f"트렌드 분석에는 최소 {min_sessions}세션이 필요합니다 (현재: {n}세션).",
            )

        # Collect all metrics
        all_metrics = set()
        for tp in self.history:
            all_metrics.update(tp.features.keys())
        all_metrics.add("injury_risk")
        all_metrics.add("overall_deviation")

        x = np.arange(n, dtype=float)
        metric_trends = {}
        improving = []
        worsening = []
        stable = []

        for metric in sorted(all_metrics):
            values = []
            for tp in self.history:
                if metric == "injury_risk":
                    values.append(tp.injury_risk)
                elif metric == "overall_deviation":
                    values.append(tp.overall_deviation)
                else:
                    values.append(tp.features.get(metric, np.nan))

            y = np.array(values)
            valid = ~np.isnan(y)
            if valid.sum() < min_sessions:
                continue

            xv, yv = x[valid], y[valid]
            slope, intercept = self._linear_fit(xv, yv)
            r_squared = self._r_squared(xv, yv, slope, intercept)

            korean_name, direction = self.METRIC_LABELS.get(
                metric, (metric, "neutral")
            )

            # Determine trend direction
            norm_slope = slope / (np.std(yv) + 1e-8)
            if abs(norm_slope) < self.TREND_THRESHOLD:
                trend_dir = "stable"
                summary = f"{korean_name}: 안정적"
                stable.append(metric)
            elif direction == "lower_better":
                if slope < 0:
                    trend_dir = "improving"
                    summary = f"{korean_name}: 개선 추세 ↓"
                    improving.append(metric)
                else:
                    trend_dir = "worsening"
                    summary = f"{korean_name}: 악화 추세 ↑"
                    worsening.append(metric)
            elif direction == "higher_better":
                if slope > 0:
                    trend_dir = "improving"
                    summary = f"{korean_name}: 개선 추세 ↑"
                    improving.append(metric)
                else:
                    trend_dir = "worsening"
                    summary = f"{korean_name}: 악화 추세 ↓"
                    worsening.append(metric)
            else:
                # Neutral: just report direction
                if abs(norm_slope) < self.TREND_THRESHOLD:
                    trend_dir = "stable"
                    summary = f"{korean_name}: 안정적"
                    stable.append(metric)
                else:
                    trend_dir = "changing"
                    arrow = "↑" if slope > 0 else "↓"
                    summary = f"{korean_name}: 변화 추세 {arrow}"
                    stable.append(metric)

            metric_trends[metric] = {
                "direction": trend_dir,
                "slope": float(slope),
                "normalized_slope": float(norm_slope),
                "r_squared": float(r_squared),
                "summary_kr": summary,
                "values": yv.tolist(),
            }

        report = self._build_trend_report(metric_trends, improving, worsening, stable, n)

        return TrendAnalysis(
            metric_trends=metric_trends,
            improving_metrics=improving,
            worsening_metrics=worsening,
            stable_metrics=stable,
            sessions_analyzed=n,
            report_kr=report,
        )

    def _linear_fit(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Simple linear regression."""
        n = len(x)
        if n < 2:
            return 0.0, float(y[0]) if n == 1 else 0.0
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)
        slope = ss_xy / (ss_xx + 1e-8)
        intercept = y_mean - slope * x_mean
        return float(slope), float(intercept)

    def _r_squared(self, x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
        """Coefficient of determination."""
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot < 1e-8:
            return 1.0
        return float(1 - ss_res / ss_tot)

    def _build_trend_report(
        self,
        trends: dict,
        improving: list,
        worsening: list,
        stable: list,
        n_sessions: int,
    ) -> str:
        """Build Korean trend report."""
        lines = [
            "=" * 60,
            "  종단 보행 트렌드 분석",
            "=" * 60,
            f"  분석 세션 수: {n_sessions}",
            "",
        ]

        if improving:
            lines.append("  ✦ 개선 추세 항목:")
            for m in improving:
                t = trends[m]
                lines.append(f"    → {t['summary_kr']} (R²={t['r_squared']:.2f})")
            lines.append("")

        if worsening:
            lines.append("  ✦ 악화 추세 항목:")
            for m in worsening:
                t = trends[m]
                lines.append(f"    → {t['summary_kr']} (R²={t['r_squared']:.2f})")
            lines.append("")

        if stable:
            lines.append("  ✦ 안정 항목:")
            for m in stable:
                if m in trends:
                    lines.append(f"    → {trends[m]['summary_kr']}")
            lines.append("")

        # Overall assessment
        if worsening and not improving:
            lines.append("  종합: 주의가 필요합니다. 악화 추세 항목에 대한 관리가 필요합니다.")
        elif improving and not worsening:
            lines.append("  종합: 좋은 추세입니다! 현재 관리 방향을 유지하세요.")
        elif improving and worsening:
            lines.append("  종합: 일부 개선, 일부 악화가 보입니다. 악화 항목에 집중 관리가 필요합니다.")
        else:
            lines.append("  종합: 전체적으로 안정적인 패턴을 유지하고 있습니다.")

        lines.append("=" * 60)
        return "\n".join(lines)
