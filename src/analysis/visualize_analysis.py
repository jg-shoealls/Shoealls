"""개인 맞춤형 보행 분석 시각화 모듈.

족저압 분석, 부상 위험, 개인 프로필, 트렌드 추적 결과를
보고서 수준의 한글 차트로 시각화합니다.
"""

from pathlib import Path
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from .foot_zones import FootZoneAnalyzer, ZONE_DEFINITIONS, REGION_GROUPS
from .gait_profile import PersonalGaitProfiler, DeviationReport
from .injury_risk import InjuryRiskEngine, InjuryRiskReport
from .feedback import CorrektiveFeedbackGenerator, PersonalizedFeedback
from .trend_tracker import LongitudinalTrendTracker, TrendAnalysis

# ── 한글 폰트 설정 ────────────────────────────────────────────────────
_FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf"
_FONT_PROP = fm.FontProperties(fname=_FONT_PATH)
_FONT_PROP_LIGHT = fm.FontProperties(
    fname="/usr/share/fonts/truetype/nanum/NanumSquareR.ttf"
)
plt.rcParams["font.family"] = "NanumSquare"
plt.rcParams["axes.unicode_minus"] = False

# ── 색상 팔레트 ──────────────────────────────────────────────────────
C_PRIMARY = "#1B3A5C"
C_ACCENT = "#E8792B"
C_SUCCESS = "#2E8B57"
C_DANGER = "#C0392B"
C_WARNING = "#F39C12"
C_LIGHT_BG = "#F7F9FC"
C_INFO = "#2196F3"

RISK_COLORS = {
    "정상": C_SUCCESS,
    "주의": C_WARNING,
    "경고": C_ACCENT,
    "위험": C_DANGER,
}

ZONE_CMAP = LinearSegmentedColormap.from_list(
    "pressure", ["#FFFFFF", "#FFF3E0", "#FF9800", "#E65100", "#B71C1C"]
)


def _set_ax_style(ax, title="", xlabel="", ylabel=""):
    """공통 축 스타일."""
    if title:
        ax.set_title(title, fontproperties=_FONT_PROP, fontsize=13, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=_FONT_PROP_LIGHT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=_FONT_PROP_LIGHT, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15, linewidth=0.5)


def plot_pressure_heatmap(
    pressure_2d: np.ndarray,
    save_path: Path,
    title: str = "족저압 분포",
):
    """단일 프레임 족저압 히트맵 + 영역 경계선."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="white")

    # 좌: 원본 히트맵
    ax = axes[0]
    if pressure_2d.ndim == 3:
        pressure_2d = pressure_2d[0]
    im = ax.imshow(pressure_2d, cmap=ZONE_CMAP, aspect="auto", interpolation="bilinear")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="압력 (상대값)")
    _set_ax_style(ax, "족저압 분포 (원본)", "좌우 (Lateral)", "전후 (Ant-Post)")

    # 우: 영역별 구분 + 평균 압력 표시
    ax = axes[1]
    ax.imshow(pressure_2d, cmap=ZONE_CMAP, aspect="auto", alpha=0.3, interpolation="bilinear")

    zone_colors = ["#E53935", "#FF9800", "#FFC107", "#4CAF50", "#8BC34A", "#2196F3", "#3F51B5"]
    for idx, (name, zdef) in enumerate(ZONE_DEFINITIONS.items()):
        r0, r1 = zdef["rows"]
        c0, c1 = zdef["cols"]
        rect = mpatches.Rectangle(
            (c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
            linewidth=2, edgecolor=zone_colors[idx], facecolor=zone_colors[idx], alpha=0.2,
        )
        ax.add_patch(rect)

        zone_val = pressure_2d[r0:r1, c0:c1].mean()
        cx = (c0 + c1) / 2 - 0.5
        cy = (r0 + r1) / 2 - 0.5
        ax.text(cx, cy, f"{zdef['description']}\n{zone_val:.2f}",
                ha="center", va="center", fontsize=8,
                fontproperties=_FONT_PROP_LIGHT, color=C_PRIMARY,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    _set_ax_style(ax, "해부학적 영역별 평균 압력", "좌우 (Lateral)", "전후 (Ant-Post)")

    fig.suptitle(title, fontproperties=_FONT_PROP, fontsize=16, y=1.02, color=C_PRIMARY)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_cop_trajectory(
    cop_trajectory: np.ndarray,
    save_path: Path,
):
    """COP 궤적 시각화."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")

    # 2D 궤적
    ax = axes[0]
    colors = np.linspace(0, 1, len(cop_trajectory))
    scatter = ax.scatter(cop_trajectory[:, 0], cop_trajectory[:, 1],
                         c=colors, cmap="coolwarm", s=15, alpha=0.7, zorder=2)
    ax.plot(cop_trajectory[:, 0], cop_trajectory[:, 1],
            "-", color="#AAA", linewidth=0.5, alpha=0.5, zorder=1)
    ax.scatter(cop_trajectory[0, 0], cop_trajectory[0, 1],
               marker="^", s=100, color=C_INFO, zorder=3, label="시작")
    ax.scatter(cop_trajectory[-1, 0], cop_trajectory[-1, 1],
               marker="s", s=100, color=C_DANGER, zorder=3, label="끝")
    fig.colorbar(scatter, ax=ax, label="시간 (프레임)")
    _set_ax_style(ax, "체중심 (COP) 궤적", "좌우 위치", "전후 위치")
    ax.legend(prop=_FONT_PROP_LIGHT, fontsize=9)

    # X축 시계열
    ax = axes[1]
    frames = np.arange(len(cop_trajectory))
    ax.plot(frames, cop_trajectory[:, 0], "-", color=C_INFO, linewidth=1.5)
    ax.fill_between(frames, cop_trajectory[:, 0], alpha=0.1, color=C_INFO)
    ax.axhline(y=0.5, color="#CCC", linestyle=":", linewidth=1)
    _set_ax_style(ax, "좌우 COP 변화", "프레임", "좌우 위치 (0=내측, 1=외측)")

    # Y축 시계열
    ax = axes[2]
    ax.plot(frames, cop_trajectory[:, 1], "-", color=C_ACCENT, linewidth=1.5)
    ax.fill_between(frames, cop_trajectory[:, 1], alpha=0.1, color=C_ACCENT)
    ax.axhline(y=0.5, color="#CCC", linestyle=":", linewidth=1)
    _set_ax_style(ax, "전후 COP 변화", "프레임", "전후 위치 (0=발끝, 1=뒤꿈치)")

    fig.suptitle("체중심(COP) 궤적 분석", fontproperties=_FONT_PROP, fontsize=16, y=1.02, color=C_PRIMARY)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_zone_temporal(
    zone_temporal: dict,
    save_path: Path,
):
    """영역별 시계열 압력 분석."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    zones = list(zone_temporal.keys())
    zone_kr = [ZONE_DEFINITIONS[z]["description"] for z in zones]

    # 평균 압력 바차트
    ax = axes[0]
    means = [zone_temporal[z]["mean_pressure_avg"] for z in zones]
    stds = [zone_temporal[z]["mean_pressure_std"] for z in zones]
    colors = ["#E53935", "#FF9800", "#FFC107", "#4CAF50", "#8BC34A", "#2196F3", "#3F51B5"]

    bars = ax.barh(range(len(zones)), means, xerr=stds, color=colors, alpha=0.8,
                   edgecolor="white", linewidth=1.5, capsize=3)
    for bar, m in zip(bars, means):
        ax.text(m + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{m:.3f}", va="center", fontsize=10, fontproperties=_FONT_PROP_LIGHT)

    ax.set_yticks(range(len(zones)))
    ax.set_yticklabels(zone_kr, fontproperties=_FONT_PROP_LIGHT, fontsize=10)
    _set_ax_style(ax, "영역별 평균 압력 (±표준편차)", "평균 압력", "")
    ax.invert_yaxis()

    # 최고 압력 바차트
    ax = axes[1]
    peak_maxs = [zone_temporal[z]["peak_pressure_max"] for z in zones]
    peak_avgs = [zone_temporal[z]["peak_pressure_avg"] for z in zones]

    x = np.arange(len(zones))
    w = 0.35
    bars1 = ax.barh(x - w / 2, peak_avgs, w, color=colors, alpha=0.6,
                     edgecolor="white", label="평균 최고 압력")
    bars2 = ax.barh(x + w / 2, peak_maxs, w, color=colors, alpha=0.9,
                     edgecolor="white", label="최대 최고 압력")

    ax.set_yticks(range(len(zones)))
    ax.set_yticklabels(zone_kr, fontproperties=_FONT_PROP_LIGHT, fontsize=10)
    _set_ax_style(ax, "영역별 최고 압력", "최고 압력", "")
    ax.legend(prop=_FONT_PROP_LIGHT, fontsize=9)
    ax.invert_yaxis()

    fig.suptitle("해부학적 영역별 압력 분석", fontproperties=_FONT_PROP, fontsize=16, y=1.02, color=C_PRIMARY)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_injury_risk_dashboard(
    injury_report: InjuryRiskReport,
    save_path: Path,
):
    """부상 위험도 대시보드."""
    fig = plt.figure(figsize=(18, 10), facecolor="white")

    fig.text(0.03, 0.97, "부상 위험 평가 리포트",
             fontproperties=_FONT_PROP, fontsize=20, va="top", color=C_PRIMARY)
    fig.text(0.03, 0.935, f"평가일: {date.today().strftime('%Y-%m-%d')}  |  전체 위험도: {injury_report.overall_risk:.0%}",
             fontproperties=_FONT_PROP_LIGHT, fontsize=11, va="top", color="#666")

    line = plt.Line2D([0.03, 0.97], [0.92, 0.92], color=C_PRIMARY, linewidth=2,
                       transform=fig.transFigure)
    fig.add_artist(line)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.06, right=0.94, top=0.88, bottom=0.06,
                           hspace=0.4, wspace=0.35)

    risks = sorted(injury_report.risks, key=lambda r: -r.risk_score)

    # ── 레이더 차트 (좌상) ──
    ax = fig.add_subplot(gs[0, 0], polar=True)
    labels = [r.korean_name for r in risks]
    values = [r.risk_score for r in risks]
    num = len(labels)
    angles = np.linspace(0, 2 * np.pi, num, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    ax.fill(angles_plot, values_plot, color=C_DANGER, alpha=0.15)
    ax.plot(angles_plot, values_plot, "o-", color=C_DANGER, linewidth=2, markersize=6)

    # 기준선
    ref_vals = [0.25] * (num + 1)
    ax.plot(angles_plot, ref_vals, "--", color=C_WARNING, linewidth=1, alpha=0.7)
    ref_vals2 = [0.5] * (num + 1)
    ax.plot(angles_plot, ref_vals2, "--", color=C_DANGER, linewidth=1, alpha=0.7)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontproperties=_FONT_PROP_LIGHT, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["주의", "경고", "위험", ""], fontsize=7)
    ax.set_title("부상 위험 레이더", fontproperties=_FONT_PROP, fontsize=13, pad=20)

    # ── 수평 바 차트 (우상) ──
    ax = fig.add_subplot(gs[0, 1])
    bar_colors = [RISK_COLORS.get(r.severity, "#999") for r in risks]
    bars = ax.barh(range(len(risks)), [r.risk_score for r in risks],
                   color=bar_colors, alpha=0.85, edgecolor="white", height=0.6)

    for i, (bar, risk) in enumerate(zip(bars, risks)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{risk.risk_score:.0%} ({risk.severity})",
                va="center", fontsize=10, fontproperties=_FONT_PROP_LIGHT,
                color=RISK_COLORS.get(risk.severity, "#999"))

    # 경계선
    ax.axvline(x=0.25, color=C_WARNING, linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=0.50, color=C_ACCENT, linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=0.75, color=C_DANGER, linestyle="--", linewidth=1, alpha=0.5)

    ax.set_yticks(range(len(risks)))
    ax.set_yticklabels([r.korean_name for r in risks],
                       fontproperties=_FONT_PROP_LIGHT, fontsize=11)
    _set_ax_style(ax, "부상 유형별 위험 점수", "위험도 (0=안전, 1=위험)", "")
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()

    # ── 위험 요인 상세 (하단 전체) ──
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    _set_ax_style(ax, "위험 요인 및 권장사항 상세")

    col_labels = ["부상 유형", "위험도", "등급", "위험 요인", "권장사항"]
    table_data = []
    for risk in risks:
        table_data.append([
            risk.korean_name,
            f"{risk.risk_score:.0%}",
            risk.severity,
            ", ".join(risk.contributing_factors[:2]),
            risk.recommendation[:30] + "..." if len(risk.recommendation) > 30 else risk.recommendation,
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colWidths=[0.12, 0.08, 0.07, 0.35, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDD")
        if row == 0:
            cell.set_facecolor(C_PRIMARY)
            cell.set_text_props(color="white", fontproperties=_FONT_PROP, fontsize=9)
        else:
            cell.set_facecolor("white")
            cell.set_text_props(fontproperties=_FONT_PROP_LIGHT, fontsize=9)
            # 등급별 색상
            if col == 2:
                severity = table_data[row - 1][2]
                cell.set_facecolor({
                    "정상": "#E8F5E9", "주의": "#FFF3E0",
                    "경고": "#FBE9E7", "위험": "#FFEBEE",
                }.get(severity, "white"))

    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_gait_profile_deviation(
    deviation_report: DeviationReport,
    session_features: dict,
    baseline_means: dict,
    save_path: Path,
):
    """개인 기준 대비 편차 시각화."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")

    metrics = sorted(deviation_report.deviations.keys())
    if not metrics:
        # 편차 데이터가 없으면 간단한 메시지만
        fig.text(0.5, 0.5, "아직 기준선이 충분하지 않습니다\n(최소 2세션 필요)",
                 ha="center", va="center", fontproperties=_FONT_PROP, fontsize=16, color="#999")
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        return

    z_scores = [deviation_report.deviations[m] for m in metrics]

    METRIC_KR = {
        "ml_index": "내외측 분포",
        "ap_index": "전후방 분포",
        "arch_index": "아치 지수",
        "cop_sway": "COP 흔들림",
        "cadence": "보행 속도",
        "stride_regularity": "보폭 규칙성",
        "step_symmetry": "좌우 대칭",
        "acceleration_rms": "가속도 크기",
    }
    metric_labels = [METRIC_KR.get(m, m) for m in metrics]

    # ── Z-score 바 차트 ──
    ax = axes[0]
    colors = []
    for z in z_scores:
        if z >= 3.0:
            colors.append(C_DANGER)
        elif z >= 2.0:
            colors.append(C_ACCENT)
        elif z >= 1.5:
            colors.append(C_WARNING)
        else:
            colors.append(C_SUCCESS)

    bars = ax.barh(range(len(metrics)), z_scores, color=colors, alpha=0.85,
                   edgecolor="white", height=0.6)

    for bar, z in zip(bars, z_scores):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"z={z:.1f}", va="center", fontsize=10, fontproperties=_FONT_PROP_LIGHT)

    ax.axvline(x=1.5, color=C_WARNING, linestyle="--", linewidth=1, alpha=0.7, label="경미")
    ax.axvline(x=2.0, color=C_ACCENT, linestyle="--", linewidth=1, alpha=0.7, label="주의")
    ax.axvline(x=3.0, color=C_DANGER, linestyle="--", linewidth=1, alpha=0.7, label="심각")

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metric_labels, fontproperties=_FONT_PROP_LIGHT, fontsize=10)
    _set_ax_style(ax, "개인 기준 대비 편차 (Z-Score)", "표준편차 (Z)", "")
    ax.legend(prop=_FONT_PROP_LIGHT, fontsize=9, loc="lower right")
    ax.invert_yaxis()

    # ── 현재 값 vs 기준 비교 ──
    ax = axes[1]
    current_vals = []
    baseline_vals = []
    plot_metrics = []
    plot_labels = []

    for m in metrics:
        if m in session_features and m in baseline_means:
            plot_metrics.append(m)
            plot_labels.append(METRIC_KR.get(m, m))
            current_vals.append(session_features[m])
            baseline_vals.append(baseline_means[m])

    if plot_metrics:
        x = np.arange(len(plot_metrics))
        w = 0.35
        ax.barh(x - w / 2, baseline_vals, w, color=C_PRIMARY, alpha=0.6,
                edgecolor="white", label="개인 기준 (평균)")
        ax.barh(x + w / 2, current_vals, w, color=C_ACCENT, alpha=0.8,
                edgecolor="white", label="현재 세션")

        ax.set_yticks(x)
        ax.set_yticklabels(plot_labels, fontproperties=_FONT_PROP_LIGHT, fontsize=10)
        ax.legend(prop=_FONT_PROP_LIGHT, fontsize=9, loc="lower right")
        ax.invert_yaxis()

    _set_ax_style(ax, "현재 세션 vs 개인 기준", "측정값", "")

    fig.suptitle("개인 보행 프로필 편차 분석", fontproperties=_FONT_PROP, fontsize=16, y=1.02, color=C_PRIMARY)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_trend_dashboard(
    trend: TrendAnalysis,
    tracker: LongitudinalTrendTracker,
    save_path: Path,
):
    """종단 트렌드 대시보드."""
    if trend.sessions_analyzed < 3:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
        ax.text(0.5, 0.5, f"트렌드 분석에는 최소 3세션이 필요합니다\n(현재: {trend.sessions_analyzed}세션)",
                ha="center", va="center", fontproperties=_FONT_PROP, fontsize=16, color="#999",
                transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        return

    # Pick key metrics to plot
    key_metrics = ["cop_sway", "stride_regularity", "step_symmetry",
                   "ml_index", "injury_risk", "overall_deviation"]
    available = [m for m in key_metrics if m in trend.metric_trends]

    n_plots = len(available)
    if n_plots == 0:
        return

    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig = plt.figure(figsize=(6 * ncols, 5 * nrows + 1.5), facecolor="white")

    fig.text(0.03, 0.98, "종단 보행 트렌드 분석",
             fontproperties=_FONT_PROP, fontsize=20, va="top", color=C_PRIMARY)
    fig.text(0.03, 0.955, f"분석 세션 수: {trend.sessions_analyzed}  |  기간 추이 분석",
             fontproperties=_FONT_PROP_LIGHT, fontsize=11, va="top", color="#666")

    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           left=0.06, right=0.94, top=0.90, bottom=0.06,
                           hspace=0.4, wspace=0.3)

    METRIC_KR = {
        "cop_sway": "체중심 흔들림",
        "stride_regularity": "보폭 규칙성",
        "step_symmetry": "좌우 대칭성",
        "ml_index": "내외측 분포",
        "injury_risk": "부상 위험도",
        "overall_deviation": "개인 기준 편차",
    }

    for idx, metric in enumerate(available):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[row, col])

        t = trend.metric_trends[metric]
        values = t["values"]
        sessions = np.arange(1, len(values) + 1)

        # Determine trend color
        direction = t["direction"]
        if direction == "improving":
            color = C_SUCCESS
            marker = "v"
        elif direction == "worsening":
            color = C_DANGER
            marker = "^"
        else:
            color = C_INFO
            marker = "o"

        # Data points
        ax.plot(sessions, values, f"{marker}-", color=color, linewidth=2, markersize=8, alpha=0.9)
        ax.fill_between(sessions, values, alpha=0.08, color=color)

        # Trend line
        slope = t["slope"]
        x_fit = np.array([sessions[0], sessions[-1]])
        y_fit = slope * (x_fit - 1) + values[0]
        ax.plot(x_fit, y_fit, "--", color=color, linewidth=1.5, alpha=0.5)

        # Annotation
        kr_name = METRIC_KR.get(metric, metric)
        direction_kr = {"improving": "개선", "worsening": "악화", "stable": "안정", "changing": "변화"}.get(direction, "")
        r2 = t["r_squared"]

        _set_ax_style(ax, f"{kr_name}", "세션", "측정값")
        ax.text(0.02, 0.98, f"{direction_kr} (R\u00b2={r2:.2f})",
                transform=ax.transAxes, fontsize=10, fontproperties=_FONT_PROP,
                va="top", color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=color))

        ax.set_xticks(sessions)

    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_full_analysis_report(
    pressure_seq: np.ndarray,
    imu_seq: np.ndarray | None,
    save_dir: Path,
    session_label: str = "세션 1",
    profiler: PersonalGaitProfiler | None = None,
    tracker: LongitudinalTrendTracker | None = None,
):
    """전체 분석 결과를 5페이지 리포트로 생성.

    Args:
        pressure_seq: (T, 1, H, W) or (T, H, W)
        imu_seq: Optional (C, T) IMU data
        save_dir: 저장 디렉토리
        session_label: 세션 이름
        profiler: 기존 개인 프로파일러 (있으면 편차 분석 포함)
        tracker: 기존 트렌드 트래커 (있으면 종단 분석 포함)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    analyzer = FootZoneAnalyzer()
    injury_engine = InjuryRiskEngine()
    feedback_gen = CorrektiveFeedbackGenerator()

    if profiler is None:
        profiler = PersonalGaitProfiler()
    if tracker is None:
        tracker = LongitudinalTrendTracker()

    # 1. Analyze
    seq_analysis = analyzer.analyze_sequence(pressure_seq)
    features = profiler.extract_session_features(pressure_seq, imu_seq)
    profiler.update_baseline(features)
    deviation = profiler.compute_deviations(features)
    injury_report = injury_engine.assess_risk(pressure_seq)
    tracker.add_session(features, injury_report.overall_risk, deviation.overall_deviation)
    feedback = feedback_gen.generate(injury_report, deviation, profiler.baseline)
    trend = tracker.analyze_trends(min_sessions=3)

    # 2. Generate all plots
    # 평균 프레임 히트맵
    if pressure_seq.ndim == 4:
        avg_frame = pressure_seq[:, 0].mean(axis=0)
    elif pressure_seq.ndim == 3:
        avg_frame = pressure_seq.mean(axis=0)
    else:
        avg_frame = pressure_seq

    plot_pressure_heatmap(avg_frame, save_dir / "analysis_p1_pressure.png",
                         title=f"족저압 분석 - {session_label}")

    plot_cop_trajectory(seq_analysis["cop_trajectory"],
                       save_dir / "analysis_p2_cop.png")

    plot_zone_temporal(seq_analysis["zone_temporal"],
                      save_dir / "analysis_p3_zones.png")

    plot_injury_risk_dashboard(injury_report,
                               save_dir / "analysis_p4_injury.png")

    # 편차 분석 (기준선 충분할 때)
    baseline_means = {}
    if profiler.baseline and profiler.baseline.num_sessions >= 2:
        for attr in ["ml_index", "ap_index", "arch_index", "cop_sway",
                     "cadence", "stride_regularity", "step_symmetry", "acceleration_rms"]:
            val = getattr(profiler.baseline, attr, (0, 0))
            if isinstance(val, tuple):
                baseline_means[attr] = val[0]

    plot_gait_profile_deviation(deviation, features, baseline_means,
                                save_dir / "analysis_p5_deviation.png")

    plot_trend_dashboard(trend, tracker, save_dir / "analysis_p6_trend.png")

    print(f"\n분석 시각화 생성 완료 ({session_label}):")
    for f in sorted(save_dir.glob("analysis_*.png")):
        print(f"  {f}")

    return {
        "features": features,
        "deviation": deviation,
        "injury_report": injury_report,
        "feedback": feedback,
        "trend": trend,
        "profiler": profiler,
        "tracker": tracker,
    }
