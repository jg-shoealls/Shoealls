"""연구소 팀 회의 보고용 시각화 모듈.

한글 레이블, 체계적 레이아웃, 정량 지표 요약 테이블을 포함한
발표/보고 수준의 시각화를 생성합니다.
"""

import platform
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

# ── 한글 폰트 설정 (OS별 호환성 유지) ──────────────────────────────────
def _setup_fonts():
    system_res = platform.system()
    if system_res == "Windows":
        font_names = ["Malgun Gothic", "NanumSquare", "NanumGothic", "Dotum", "Gulim"]
    elif system_res == "Darwin":
        font_names = ["AppleGothic", "NanumSquare", "NanumGothic"]
    else:
        font_names = ["NanumSquare", "NanumGothic", "NanumBarunGothic", "UnDotum"]

    selected_font = "sans-serif"
    for font_name in font_names:
        if any(font_name in f.name for f in fm.fontManager.ttflist):
            selected_font = font_name
            break
    
    return fm.FontProperties(family=selected_font, weight="bold"), \
           fm.FontProperties(family=selected_font, weight="normal"), \
           selected_font

_FONT_PROP, _FONT_PROP_LIGHT, _FONT_NAME = _setup_fonts()
plt.rcParams["font.family"] = _FONT_NAME
plt.rcParams["axes.unicode_minus"] = False

# ── 색상 팔레트 (보고서용 톤, 12개 이상 클래스 대응) ─────────────────────
C_PRIMARY = "#1B3A5C"       # 진한 남색
C_ACCENT = "#E8792B"        # 주황 강조
C_SUCCESS = "#2E8B57"       # 초록 (성공)
C_DANGER = "#C0392B"        # 빨강 (실패/경고)
C_LIGHT_BG = "#F7F9FC"      # 밝은 배경

CLASS_COLORS = [
    "#2196F3", "#FF9800", "#E53935", "#8E24AA", 
    "#009688", "#795548", "#607D8B", "#FFC107",
    "#9C27B0", "#00BCD4", "#4CAF50", "#FF5722"
]
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
    """연구소 보고용 전체 시각화 리포트 생성."""
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = _compute_all_metrics(y_true, y_pred, probs, class_names)

    _page1_summary(history, metrics, class_names, model_params, save_dir)
    _page2_detail(metrics, class_names, save_dir)
    _page3_ablation(ablation_results, len(class_names), save_dir)

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

    fig.text(0.03, 0.97, "멀티모달 보행 데이터 기반 AI 알고리즘",
             fontproperties=_FONT_PROP, fontsize=22, va="top", color=C_PRIMARY)
    fig.text(0.03, 0.935, "초기 검증 결과 보고  |  합성 데이터 기반 Proof-of-Concept",
             fontproperties=_FONT_PROP_LIGHT, fontsize=12, va="top", color="#666")
    fig.text(0.97, 0.97, f"보고일: {date.today().strftime('%Y-%m-%d')}",
             fontproperties=_FONT_PROP_LIGHT, fontsize=10, va="top", ha="right", color="#999")

    line = plt.Line2D([0.03, 0.97], [0.92, 0.92], color=C_PRIMARY, linewidth=2,
                       transform=fig.transFigure)
    fig.add_artist(line)

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           left=0.05, right=0.95, top=0.88, bottom=0.06,
                           hspace=0.35, wspace=0.35)

    ax_kpi = fig.add_subplot(gs[0, 0])
    ax_kpi.axis("off")
    kpi_data = [
        ("정확도", f"{metrics['accuracy']:.1%}", C_SUCCESS),
        ("F1 Score", f"{metrics['f1_macro']:.4f}", C_PRIMARY),
        ("학습 에포크", f"{len(history['train_loss'])}", C_ACCENT),
        ("모델 파라미터", f"{model_params:,}", "#666"),
    ]

    for i, (label, value, color) in enumerate(kpi_data):
        y = 0.88 - i * 0.25
        rect = mpatches.FancyBboxPatch(
            (0.02, y - 0.08), 0.96, 0.22,
            boxstyle="round,pad=0.02", facecolor=C_LIGHT_BG,
            edgecolor="#DDD", linewidth=0.5,
        )
        ax_kpi.add_patch(rect)
        ax_kpi.text(0.08, y + 0.05, label, fontproperties=_FONT_PROP_LIGHT, fontsize=9, color="#888", va="center")
        ax_kpi.text(0.08, y - 0.02, value, fontproperties=_FONT_PROP, fontsize=16, color=color, va="center", fontweight="bold")

    # Loss & Accuracy Curves
    for idx, (key_prefix, title) in enumerate([("loss", "손실 함수 (Loss)"), ("acc", "분류 정확도")]):
        ax = fig.add_subplot(gs[0, idx + 1])
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history[f"train_{key_prefix}"], "-", label="학습", color=C_PRIMARY, linewidth=2)
        ax.plot(epochs, history[f"val_{key_prefix}"], "--", label="검증", color=C_ACCENT, linewidth=2)
        _set_ax_style(ax, title, "에포크", key_prefix.capitalize())
        if key_prefix == "loss": ax.set_yscale("log")
        else: ax.set_ylim(0, 1.05)
        ax.legend(prop=_FONT_PROP_LIGHT, fontsize=9)

    # Confusion Matrix
    ax = fig.add_subplot(gs[0, 3])
    cm = metrics["cm"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums > 0, out=np.zeros_like(cm, dtype=float))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    kr_names = [_kr(n) for n in class_names]
    short_kr = [n[:6] + ".." if len(n) > 7 else n for n in kr_names]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(short_kr, fontproperties=_FONT_PROP_LIGHT, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(short_kr, fontproperties=_FONT_PROP_LIGHT, fontsize=7)
    _set_ax_style(ax, "혼동 행렬", "예측", "실제")

    # Table
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")
    col_labels = ["보행 패턴", "샘플 수", "Precision", "Recall", "F1 Score", "분류 정확도", "평균 신뢰도"]
    table_data = []
    for i, name in enumerate(class_names):
        mask = metrics["y_true"] == i
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        avg_conf = metrics["max_probs"][mask].mean() if mask.any() else 0
        table_data.append([_kr(name), f"{metrics['support'][i]}", f"{metrics['precision'][i]:.4f}", f"{metrics['recall'][i]:.4f}", f"{metrics['f1'][i]:.4f}", f"{class_acc:.1%}", f"{avg_conf:.4f}"])
    
    table_data.append(["전체 (Macro Avg)", f"{int(metrics['support'].sum())}", f"{metrics['precision'].mean():.4f}", f"{metrics['recall'].mean():.4f}", f"{metrics['f1'].mean():.4f}", f"{metrics['accuracy']:.1%}", f"{metrics['max_probs'].mean():.4f}"])
    
    table = ax_table.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.0, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_facecolor(C_PRIMARY); cell.set_text_props(color="white", fontproperties=_FONT_PROP)
        else: cell.set_text_props(fontproperties=_FONT_PROP_LIGHT)

    fig.savefig(save_dir / "report_p1_summary.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 2: 상세 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _page2_detail(metrics, class_names, save_dir):
    fig = plt.figure(figsize=(22, 12), facecolor="white")
    fig.text(0.03, 0.97, "상세 성능 분석", fontproperties=_FONT_PROP, fontsize=20, va="top", color=C_PRIMARY)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.06, right=0.94, top=0.88, bottom=0.08, hspace=0.35, wspace=0.3)

    kr_names = [_kr(n) for n in class_names]
    short_kr = [n[:6] + ".." if len(n) > 7 else n for n in kr_names]
    x = np.arange(len(class_names))

    # Bar chart for P/R/F1
    ax = fig.add_subplot(gs[0, 0])
    w = 0.25
    ax.bar(x - w, metrics["precision"], w, label="Precision", color=C_PRIMARY, alpha=0.8)
    ax.bar(x, metrics["recall"], w, label="Recall", color=C_ACCENT, alpha=0.8)
    ax.bar(x + w, metrics["f1"], w, label="F1", color=C_SUCCESS, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(short_kr, fontproperties=_FONT_PROP_LIGHT, fontsize=8, rotation=30)
    _set_ax_style(ax, "클래스별 분류 성능")
    ax.legend(prop=_FONT_PROP_LIGHT)

    # Boxplot for confidence
    ax = fig.add_subplot(gs[1, 1])
    class_confs = [metrics["max_probs"][metrics["y_true"] == i] for i in range(len(class_names))]
    bp = ax.boxplot(class_confs, labels=short_kr, patch_artist=True)
    for patch, color in zip(bp["boxes"], CLASS_COLORS * 2):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    _set_ax_style(ax, "클래스별 예측 신뢰도")
    for label in ax.get_xticklabels(): label.set_fontproperties(_FONT_PROP_LIGHT); label.set_rotation(30)

    fig.savefig(save_dir / "report_p2_detail.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 3: 모달리티 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _page3_ablation(ablation_results, num_classes, save_dir):
    fig = plt.figure(figsize=(22, 12), facecolor="white")
    fig.text(0.03, 0.97, "모달리티 기여도 분석", fontproperties=_FONT_PROP, fontsize=20, va="top", color=C_PRIMARY)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.06, right=0.94, top=0.88, bottom=0.08, hspace=0.4, wspace=0.3)

    names_en = list(ablation_results.keys())
    names_kr = [MODALITY_KR.get(n, n) for n in names_en]
    accs = list(ablation_results.values())

    ax = fig.add_subplot(gs[0, :])
    ax.barh(range(len(names_kr)), accs, color=C_PRIMARY, alpha=0.7)
    ax.set_yticks(range(len(names_kr))); ax.set_yticklabels(names_kr, fontproperties=_FONT_PROP_LIGHT)
    _set_ax_style(ax, "센서 조합별 분류 정확도")
    ax.invert_yaxis()

    # Architecture simplified
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, f"모델 구조: Multimodal Fusion\n최종 분류: {num_classes} 클래스", 
            ha="center", va="center", fontproperties=_FONT_PROP, fontsize=15, bbox=dict(facecolor=C_LIGHT_BG, alpha=0.5))

    fig.savefig(save_dir / "report_p3_ablation.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
