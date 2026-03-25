"""연구소 팀 회의 보고용 시각화 모듈.

한글 레이블, 체계적 레이아웃, 정량 지표 요약 테이블을 포함한
발표/보고 수준의 시각화를 생성합니다.
"""

from pathlib import Path
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
)

# ── 한글 폰트 설정 ────────────────────────────────────────────────────
_FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf"
_FONT_PROP = fm.FontProperties(fname=_FONT_PATH)
_FONT_PROP_LIGHT = fm.FontProperties(
    fname="/usr/share/fonts/truetype/nanum/NanumSquareR.ttf"
)

plt.rcParams["font.family"] = "NanumSquare"
plt.rcParams["axes.unicode_minus"] = False

# ── 색상 팔레트 (보고서용 톤) ──────────────────────────────────────────
C_PRIMARY = "#1B3A5C"       # 진한 남색
C_ACCENT = "#E8792B"        # 주황 강조
C_SUCCESS = "#2E8B57"       # 초록 (성공)
C_DANGER = "#C0392B"        # 빨강 (실패/경고)
C_LIGHT_BG = "#F7F9FC"      # 밝은 배경

CLASS_COLORS = ["#2196F3", "#FF9800", "#E53935", "#8E24AA"]
CLASS_KR = {
    "normal": "정상 보행",
    "antalgic": "절뚝거림(Antalgic)",
    "ataxic": "운동실조(Ataxic)",
    "parkinsonian": "파킨슨(Parkinsonian)",
}

MODALITY_KR = {
    "IMU only": "IMU 단독",
    "Pressure only": "족저압 단독",
    "Skeleton only": "스켈레톤 단독",
    "IMU + Pressure": "IMU + 족저압",
    "IMU + Skeleton": "IMU + 스켈레톤",
    "Pressure + Skeleton": "족저압 + 스켈레톤",
    "All (Fusion)": "전체 융합 (제안 기법)",
}


def _kr(name: str) -> str:
    return CLASS_KR.get(name, name)


def _set_ax_style(ax, title="", xlabel="", ylabel=""):
    """공통 축 스타일 설정."""
    if title:
        ax.set_title(title, fontproperties=_FONT_PROP, fontsize=13, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=_FONT_PROP_LIGHT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=_FONT_PROP_LIGHT, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15, linewidth=0.5)


def generate_report(
    history: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    ablation_results: dict,
    model_params: int,
    save_dir: Path,
):
    """연구소 보고용 전체 시각화 리포트 생성.

    총 3페이지:
      1. 요약 대시보드 (핵심 지표 + 학습곡선 + 혼동행렬)
      2. 상세 분석 (클래스별 성능 + 신뢰도 분석)
      3. 모달리티 기여도 분석 (Ablation Study + 아키텍처 다이어그램)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = _compute_all_metrics(y_true, y_pred, probs, class_names)

    _page1_summary(history, metrics, class_names, model_params, save_dir)
    _page2_detail(metrics, class_names, save_dir)
    _page3_ablation(ablation_results, save_dir)

    print(f"\n보고서 생성 완료:")
    for f in sorted(save_dir.glob("report_*.png")):
        print(f"  {f}")


def _compute_all_metrics(y_true, y_pred, probs, class_names):
    """모든 지표를 한 번에 계산."""
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0,
    )
    max_probs = probs.max(axis=1)
    correct = y_true == y_pred

    return {
        "cm": cm,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support,
        "max_probs": max_probs,
        "correct": correct,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 1: 요약 대시보드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _page1_summary(history, metrics, class_names, model_params, save_dir):
    fig = plt.figure(figsize=(22, 14), facecolor="white")

    # 상단 타이틀 영역
    fig.text(0.03, 0.97, "멀티모달 보행 데이터 기반 AI 알고리즘",
             fontproperties=_FONT_PROP, fontsize=22, va="top", color=C_PRIMARY)
    fig.text(0.03, 0.935, "초기 검증 결과 보고  |  합성 데이터 기반 Proof-of-Concept",
             fontproperties=_FONT_PROP_LIGHT, fontsize=12, va="top", color="#666")
    fig.text(0.97, 0.97, f"보고일: {date.today().strftime('%Y-%m-%d')}",
             fontproperties=_FONT_PROP_LIGHT, fontsize=10, va="top", ha="right", color="#999")

    # 구분선
    line = plt.Line2D([0.03, 0.97], [0.92, 0.92], color=C_PRIMARY, linewidth=2,
                       transform=fig.transFigure)
    fig.add_artist(line)

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           left=0.05, right=0.95, top=0.88, bottom=0.06,
                           hspace=0.35, wspace=0.35)

    # ── KPI 카드 영역 (상단 왼쪽) ──
    ax_kpi = fig.add_subplot(gs[0, 0])
    ax_kpi.set_xlim(0, 1)
    ax_kpi.set_ylim(0, 1)
    ax_kpi.axis("off")

    kpi_data = [
        ("정확도", f"{metrics['accuracy']:.1%}", C_SUCCESS),
        ("F1 Score", f"{metrics['f1_macro']:.4f}", C_PRIMARY),
        ("학습 에포크", f"{len(history['train_loss'])}", C_ACCENT),
        ("모델 파라미터", f"{model_params:,}", "#666"),
    ]

    for i, (label, value, color) in enumerate(kpi_data):
        y = 0.88 - i * 0.25
        # 카드 배경
        rect = mpatches.FancyBboxPatch(
            (0.02, y - 0.08), 0.96, 0.22,
            boxstyle="round,pad=0.02", facecolor=C_LIGHT_BG,
            edgecolor="#DDD", linewidth=0.5,
        )
        ax_kpi.add_patch(rect)
        ax_kpi.text(0.08, y + 0.05, label,
                    fontproperties=_FONT_PROP_LIGHT, fontsize=9, color="#888", va="center")
        ax_kpi.text(0.08, y - 0.02, value,
                    fontproperties=_FONT_PROP, fontsize=16, color=color, va="center",
                    fontweight="bold")

    # ── 학습 곡선 (Loss) ──
    ax = fig.add_subplot(gs[0, 1])
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "-", label="학습", color=C_PRIMARY, linewidth=2)
    ax.plot(epochs, history["val_loss"], "--", label="검증", color=C_ACCENT, linewidth=2)
    ax.fill_between(epochs, history["train_loss"], alpha=0.08, color=C_PRIMARY)
    _set_ax_style(ax, "손실 함수 (Loss)", "에포크", "Loss")
    ax.set_yscale("log")
    ax.legend(prop=_FONT_PROP_LIGHT, fontsize=9)

    # ── 학습 곡선 (Accuracy) ──
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs, history["train_acc"], "-", label="학습", color=C_PRIMARY, linewidth=2)
    ax.plot(epochs, history["val_acc"], "--", label="검증", color=C_ACCENT, linewidth=2)
    ax.fill_between(epochs, history["train_acc"], alpha=0.08, color=C_PRIMARY)
    _set_ax_style(ax, "분류 정확도", "에포크", "Accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="#CCC", linestyle=":", linewidth=0.8)
    ax.legend(prop=_FONT_PROP_LIGHT, fontsize=9)

    # ── 혼동 행렬 ──
    ax = fig.add_subplot(gs[0, 3])
    cm = metrics["cm"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums > 0,
                        out=np.zeros_like(cm, dtype=float))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    kr_names = [_kr(n) for n in class_names]
    short_kr = ["정상", "절뚝거림", "운동실조", "파킨슨"]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(short_kr, fontproperties=_FONT_PROP_LIGHT, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(short_kr, fontproperties=_FONT_PROP_LIGHT, fontsize=8)
    _set_ax_style(ax, "혼동 행렬", "예측", "실제")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.5 else C_PRIMARY
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                    ha="center", va="center", fontsize=8, fontweight="bold", color=color)

    # ── 하단: 정량 지표 요약 테이블 ──
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")
    _set_ax_style(ax_table, "클래스별 정량 지표 요약")

    col_labels = ["보행 패턴", "샘플 수", "Precision", "Recall", "F1 Score", "분류 정확도", "평균 신뢰도"]
    table_data = []
    for i, name in enumerate(class_names):
        mask = metrics["y_true"] == i
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        avg_conf = metrics["max_probs"][mask].mean() if mask.any() else 0
        table_data.append([
            _kr(name),
            f"{metrics['support'][i]}",
            f"{metrics['precision'][i]:.4f}",
            f"{metrics['recall'][i]:.4f}",
            f"{metrics['f1'][i]:.4f}",
            f"{class_acc:.1%}",
            f"{avg_conf:.4f}",
        ])

    # 매크로 평균 행
    table_data.append([
        "전체 (Macro Avg)",
        f"{int(metrics['support'].sum())}",
        f"{metrics['precision'].mean():.4f}",
        f"{metrics['recall'].mean():.4f}",
        f"{metrics['f1'].mean():.4f}",
        f"{metrics['accuracy']:.1%}",
        f"{metrics['max_probs'].mean():.4f}",
    ])

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    # 테이블 스타일
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDD")
        if row == 0:  # 헤더
            cell.set_facecolor(C_PRIMARY)
            cell.set_text_props(color="white", fontproperties=_FONT_PROP, fontsize=10)
        elif row == len(table_data):  # 마지막 행 (합계)
            cell.set_facecolor("#E8EDF2")
            cell.set_text_props(fontproperties=_FONT_PROP, fontsize=10)
        else:
            cell.set_facecolor("white")
            cell.set_text_props(fontproperties=_FONT_PROP_LIGHT, fontsize=10)
            # F1이 1.0이면 초록 배경
            if col in [2, 3, 4] and "1.00" in cell.get_text().get_text():
                cell.set_facecolor("#E8F5E9")

    fig.savefig(save_dir / "report_p1_summary.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [1/3] 요약 대시보드: report_p1_summary.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 2: 상세 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _page2_detail(metrics, class_names, save_dir):
    fig = plt.figure(figsize=(22, 12), facecolor="white")

    fig.text(0.03, 0.97, "상세 성능 분석",
             fontproperties=_FONT_PROP, fontsize=20, va="top", color=C_PRIMARY)
    fig.text(0.03, 0.94, "클래스별 성능 지표 및 예측 신뢰도 분포",
             fontproperties=_FONT_PROP_LIGHT, fontsize=11, va="top", color="#666")
    line = plt.Line2D([0.03, 0.97], [0.925, 0.925], color=C_PRIMARY, linewidth=2,
                       transform=fig.transFigure)
    fig.add_artist(line)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.06, right=0.94, top=0.88, bottom=0.08,
                           hspace=0.35, wspace=0.3)

    kr_names = [_kr(n) for n in class_names]
    short_kr = ["정상", "절뚝거림", "운동실조", "파킨슨"]

    # ── 클래스별 Precision / Recall / F1 ──
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(class_names))
    w = 0.22

    bars_p = ax.bar(x - w, metrics["precision"], w, label="Precision",
                    color=C_PRIMARY, alpha=0.85)
    bars_r = ax.bar(x, metrics["recall"], w, label="Recall",
                    color=C_ACCENT, alpha=0.85)
    bars_f = ax.bar(x + w, metrics["f1"], w, label="F1 Score",
                    color=C_SUCCESS, alpha=0.85)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(short_kr, fontproperties=_FONT_PROP_LIGHT, fontsize=10)
    _set_ax_style(ax, "클래스별 분류 성능", "", "점수")
    ax.set_ylim(0, 1.18)
    ax.legend(prop=_FONT_PROP_LIGHT, fontsize=9, loc="lower right")

    # ── 클래스별 샘플 수 + 정확도 ──
    ax = fig.add_subplot(gs[0, 1])
    cm = metrics["cm"]
    class_accs = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                  for i in range(len(class_names))]

    bars = ax.bar(x, metrics["support"], color=CLASS_COLORS, alpha=0.8, edgecolor="white")
    ax2 = ax.twinx()
    ax2.plot(x, class_accs, "D-", color=C_DANGER, markersize=8, linewidth=2, label="정확도")

    for i, (s, a) in enumerate(zip(metrics["support"], class_accs)):
        ax.text(i, s + 0.5, f"n={s}", ha="center", fontsize=9,
                fontproperties=_FONT_PROP_LIGHT)
        ax2.text(i + 0.15, a - 0.03, f"{a:.0%}", fontsize=9, color=C_DANGER,
                 fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(short_kr, fontproperties=_FONT_PROP_LIGHT, fontsize=10)
    _set_ax_style(ax, "클래스별 데이터 분포 및 정확도", "", "샘플 수")
    ax2.set_ylabel("정확도", fontproperties=_FONT_PROP_LIGHT, fontsize=10)
    ax2.set_ylim(0, 1.15)
    ax2.legend(prop=_FONT_PROP_LIGHT, fontsize=9, loc="lower right")

    # ── 예측 신뢰도 분포 (정답/오답) ──
    ax = fig.add_subplot(gs[1, 0])
    max_probs = metrics["max_probs"]
    correct = metrics["correct"]
    bins = np.linspace(0, 1, 26)

    if correct.any():
        ax.hist(max_probs[correct], bins=bins, alpha=0.75, label="정답 예측",
                color=C_SUCCESS, edgecolor="white", linewidth=0.5)
    if (~correct).any():
        ax.hist(max_probs[~correct], bins=bins, alpha=0.75, label="오답 예측",
                color=C_DANGER, edgecolor="white", linewidth=0.5)

    avg_conf = max_probs[correct].mean() if correct.any() else 0
    ax.axvline(x=avg_conf, color=C_PRIMARY, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(avg_conf - 0.02, ax.get_ylim()[1] * 0.9, f"평균: {avg_conf:.2f}",
            fontproperties=_FONT_PROP_LIGHT, fontsize=9, color=C_PRIMARY,
            ha="right")

    _set_ax_style(ax, "예측 신뢰도 분포", "신뢰도 (Softmax 확률)", "빈도")
    ax.legend(prop=_FONT_PROP_LIGHT, fontsize=9)

    # ── 클래스별 신뢰도 박스플롯 ──
    ax = fig.add_subplot(gs[1, 1])
    class_confs = [max_probs[metrics["y_true"] == i] for i in range(len(class_names))]

    bp = ax.boxplot(class_confs, labels=short_kr, patch_artist=True,
                    medianprops=dict(color=C_PRIMARY, linewidth=2))
    for patch, color in zip(bp["boxes"], CLASS_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    for i, confs in enumerate(class_confs):
        ax.text(i + 1, confs.mean() + 0.02, f"{confs.mean():.2f}",
                ha="center", fontsize=9, fontweight="bold", color=C_PRIMARY)

    for label in ax.get_xticklabels():
        label.set_fontproperties(_FONT_PROP_LIGHT)

    _set_ax_style(ax, "클래스별 예측 신뢰도", "", "신뢰도")
    ax.set_ylim(0, 1.1)

    fig.savefig(save_dir / "report_p2_detail.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [2/3] 상세 분석: report_p2_detail.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 3: 모달리티 기여도 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _page3_ablation(ablation_results, save_dir):
    fig = plt.figure(figsize=(22, 12), facecolor="white")

    fig.text(0.03, 0.97, "모달리티 기여도 분석 (Ablation Study)",
             fontproperties=_FONT_PROP, fontsize=20, va="top", color=C_PRIMARY)
    fig.text(0.03, 0.94,
             "각 센서 모달리티의 개별/조합 성능을 비교하여 멀티모달 융합의 효과를 검증",
             fontproperties=_FONT_PROP_LIGHT, fontsize=11, va="top", color="#666")
    line = plt.Line2D([0.03, 0.97], [0.925, 0.925], color=C_PRIMARY, linewidth=2,
                       transform=fig.transFigure)
    fig.add_artist(line)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.06, right=0.94, top=0.88, bottom=0.08,
                           hspace=0.4, wspace=0.3)

    names_en = list(ablation_results.keys())
    names_kr = [MODALITY_KR.get(n, n) for n in names_en]
    accs = list(ablation_results.values())

    # ── 수평 바 차트 (메인) ──
    ax = fig.add_subplot(gs[0, :])
    bar_colors = []
    for name in names_en:
        if "All" in name:
            bar_colors.append(C_SUCCESS)
        elif "+" in name:
            bar_colors.append(C_ACCENT)
        else:
            bar_colors.append("#78909C")

    bars = ax.barh(range(len(names_kr)), accs, color=bar_colors, alpha=0.88,
                   edgecolor="white", linewidth=1.5, height=0.65)

    for bar, acc, name in zip(bars, accs, names_en):
        # 값 레이블
        offset = 0.02 if acc < 0.9 else -0.08
        color = C_PRIMARY if acc < 0.9 else "white"
        ax.text(acc + offset, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}", va="center", fontsize=13,
                fontproperties=_FONT_PROP, color=color)

    ax.set_yticks(range(len(names_kr)))
    ax.set_yticklabels(names_kr, fontproperties=_FONT_PROP_LIGHT, fontsize=12)
    _set_ax_style(ax, "센서 조합별 분류 정확도", "정확도 (Accuracy)", "")
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()

    # 범례
    legend_items = [
        mpatches.Patch(color="#78909C", alpha=0.88, label="단일 모달리티"),
        mpatches.Patch(color=C_ACCENT, alpha=0.88, label="2종 조합"),
        mpatches.Patch(color=C_SUCCESS, alpha=0.88, label="전체 융합 (제안 기법)"),
    ]
    ax.legend(handles=legend_items, prop=_FONT_PROP_LIGHT, fontsize=10,
              loc="lower right")

    # ── 모달리티 기여도 (단일 → 조합 → 전체) 단계별 ──
    ax = fig.add_subplot(gs[1, 0])

    # 단일 모달리티 정확도
    single = {k: v for k, v in ablation_results.items() if "only" in k.lower()}
    pair = {k: v for k, v in ablation_results.items() if "+" in k}
    fusion_acc = ablation_results.get("All (Fusion)", 0)

    categories = ["단일 모달리티\n(최고)", "2종 조합\n(최고)", "전체 융합\n(제안 기법)"]
    values = [max(single.values()), max(pair.values()), fusion_acc]
    cat_colors = ["#78909C", C_ACCENT, C_SUCCESS]

    bars = ax.bar(range(3), values, color=cat_colors, alpha=0.85, width=0.6,
                  edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.1%}", ha="center", fontsize=13,
                fontproperties=_FONT_PROP, fontweight="bold")

    # 성능 향상 화살표
    for i in range(len(values) - 1):
        improvement = values[i + 1] - values[i]
        if improvement > 0:
            mid_x = i + 0.5
            mid_y = (values[i] + values[i + 1]) / 2
            ax.annotate(f"+{improvement:.1%}",
                       xy=(mid_x, mid_y), fontsize=10, color=C_DANGER,
                       fontproperties=_FONT_PROP, fontweight="bold",
                       ha="center")

    ax.set_xticks(range(3))
    ax.set_xticklabels(categories, fontproperties=_FONT_PROP_LIGHT, fontsize=10)
    _set_ax_style(ax, "융합 단계별 성능 향상", "", "정확도")
    ax.set_ylim(0, 1.18)

    # ── 아키텍처 다이어그램 ──
    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    _set_ax_style(ax, "모델 아키텍처 개요")

    # 입력 모달리티
    inputs = [
        (0.5, 6.5, "IMU\n(가속도/자이로)", "#78909C"),
        (0.5, 4.5, "족저압\n(압력 분포)", "#78909C"),
        (0.5, 2.5, "스켈레톤\n(관절 좌표)", "#78909C"),
    ]
    for x, y, label, color in inputs:
        rect = mpatches.FancyBboxPatch(
            (x, y - 0.6), 2.2, 1.2,
            boxstyle="round,pad=0.15", facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + 1.1, y, label, ha="center", va="center",
                fontproperties=_FONT_PROP_LIGHT, fontsize=8, color=C_PRIMARY)

    # 인코더
    encoders = [
        (3.5, 6.5, "1D-CNN\n+ BiLSTM", C_PRIMARY),
        (3.5, 4.5, "2D-CNN", C_PRIMARY),
        (3.5, 2.5, "ST-GCN", C_PRIMARY),
    ]
    for x, y, label, color in encoders:
        rect = mpatches.FancyBboxPatch(
            (x, y - 0.6), 1.8, 1.2,
            boxstyle="round,pad=0.15", facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + 0.9, y, label, ha="center", va="center",
                fontproperties=_FONT_PROP_LIGHT, fontsize=8, color=C_PRIMARY)

    # 화살표: 입력 → 인코더
    for y in [6.5, 4.5, 2.5]:
        ax.annotate("", xy=(3.5, y), xytext=(2.7, y),
                   arrowprops=dict(arrowstyle="->", color="#AAA", lw=1.5))

    # 융합 모듈
    rect = mpatches.FancyBboxPatch(
        (6.2, 3.2), 1.8, 2.6,
        boxstyle="round,pad=0.2", facecolor=C_ACCENT, alpha=0.15,
        edgecolor=C_ACCENT, linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(7.1, 4.5, "Cross-Modal\nAttention\nFusion",
            ha="center", va="center",
            fontproperties=_FONT_PROP, fontsize=8, color=C_ACCENT)

    # 화살표: 인코더 → 융합
    for y in [6.5, 4.5, 2.5]:
        ax.annotate("", xy=(6.2, 4.5), xytext=(5.3, y),
                   arrowprops=dict(arrowstyle="->", color="#AAA", lw=1.5))

    # 분류기
    rect = mpatches.FancyBboxPatch(
        (8.5, 3.8), 1.3, 1.4,
        boxstyle="round,pad=0.15", facecolor=C_SUCCESS, alpha=0.15,
        edgecolor=C_SUCCESS, linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(9.15, 4.5, "분류기\n(4클래스)",
            ha="center", va="center",
            fontproperties=_FONT_PROP, fontsize=8, color=C_SUCCESS)

    ax.annotate("", xy=(8.5, 4.5), xytext=(8.0, 4.5),
               arrowprops=dict(arrowstyle="->", color="#AAA", lw=1.5))

    fig.savefig(save_dir / "report_p3_ablation.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [3/3] 모달리티 분석: report_p3_ablation.png")
