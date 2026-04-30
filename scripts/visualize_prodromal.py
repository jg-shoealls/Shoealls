"""학습 결과 시각화 스크립트.

실행:
    python scripts/visualize_prodromal.py
    python scripts/visualize_prodromal.py --checkpoint outputs/prodromal/best_model.pt --out outputs/prodromal/result.png
"""

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

from src.data.dataset import MultimodalGaitDataset
from src.data.synthetic_prodromal import (
    generate_prodromal_dataset, CLASS_NAMES, CLASS_NAMES_EN, STAGE_PROFILES,
)
from src.data.preprocessing import preprocess_imu, preprocess_pressure, preprocess_skeleton
from src.models.multimodal_gait_net import MultimodalGaitNet
from src.utils.metrics import compute_metrics


# ── 팔레트 ─────────────────────────────────────────────────────────────────────
COLORS   = ["#22c55e", "#06b6d4", "#f59e0b", "#ef4444"]   # 정상/전임상/초기/임상
COLORS_A = ["#bbf7d0", "#cffafe", "#fef3c7", "#fee2e2"]   # 연한 버전
FONT_KR  = "Malgun Gothic" if sys.platform == "win32" else "DejaVu Sans"


# ── 유틸 ───────────────────────────────────────────────────────────────────────
def setup_korean_font():
    plt.rcParams["font.family"] = FONT_KR
    plt.rcParams["axes.unicode_minus"] = False


@torch.no_grad()
def get_all_probs_and_labels(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")
        logits = model(batch)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


@torch.no_grad()
def get_demo_probs(model, raw_data, config, device, seed=99):
    """각 단계별 대표 샘플 1개의 확률 반환."""
    data_cfg = config["data"]
    seq_len  = data_cfg["sequence_length"]
    grid     = tuple(data_cfg["pressure_grid_size"])
    joints   = data_cfg["skeleton_joints"]
    model.eval()
    rng    = np.random.default_rng(seed)
    labels = raw_data["labels"]
    results = []
    for stage in range(4):
        idxs = np.where(labels == stage)[0]
        idx  = rng.choice(idxs)
        imu  = torch.from_numpy(preprocess_imu(raw_data["imu"][idx], seq_len)).unsqueeze(0).to(device)
        pres = torch.from_numpy(preprocess_pressure(raw_data["pressure"][idx], seq_len, grid)).unsqueeze(0).to(device)
        skel = torch.from_numpy(preprocess_skeleton(raw_data["skeleton"][idx], seq_len, joints)).unsqueeze(0).to(device)
        logits = model({"imu": imu, "pressure": pres, "skeleton": skel})
        probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        results.append({"stage": stage, "idx": idx, "probs": probs, "pred": int(probs.argmax())})
    return results


# ── 1. 학습 곡선 ───────────────────────────────────────────────────────────────
def plot_training_curves(ax_loss, ax_acc, history):
    epochs = range(1, len(history["train_loss"]) + 1)

    ax_loss.plot(epochs, history["train_loss"], color="#6366f1", linewidth=2, label="학습 Loss", marker="o", markersize=3)
    ax_loss.plot(epochs, history["val_loss"],   color="#f97316", linewidth=2, label="검증 Loss", marker="s", markersize=3)
    ax_loss.set_title("학습 / 검증 Loss", fontsize=12, fontweight="bold", pad=8)
    ax_loss.set_xlabel("에폭")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(framealpha=0.8)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_xlim(1, len(epochs))

    ax_acc.plot(epochs, [v * 100 for v in history["train_acc"]], color="#6366f1", linewidth=2, label="학습 Acc", marker="o", markersize=3)
    ax_acc.plot(epochs, [v * 100 for v in history["val_acc"]],   color="#f97316", linewidth=2, label="검증 Acc", marker="s", markersize=3)
    ax_acc.axhline(y=100, color="#22c55e", linestyle="--", linewidth=1, alpha=0.6, label="100%")
    ax_acc.set_title("학습 / 검증 Accuracy", fontsize=12, fontweight="bold", pad=8)
    ax_acc.set_xlabel("에폭")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(framealpha=0.8)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_xlim(1, len(epochs))


# ── 2. 혼동 행렬 ────────────────────────────────────────────────────────────────
def plot_confusion_matrix(ax, cm, class_names):
    total_per_class = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / (total_per_class + 1e-8) * 100

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("예측 (Predicted)", fontsize=10)
    ax.set_ylabel("실제 (True)", fontsize=10)
    ax.set_title("혼동 행렬 (%)", fontsize=12, fontweight="bold", pad=8)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val   = cm[i, j]
            pct   = cm_pct[i, j]
            color = "white" if pct > 60 else "black"
            ax.text(j, i, f"{int(val)}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if i == j else "normal")

    for i in range(len(class_names)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (i - 0.5, i - 0.5), 1, 1,
            boxstyle="square,pad=0", linewidth=2.5,
            edgecolor=COLORS[i], facecolor="none", zorder=3,
        ))

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ── 3. 단계별 성능 바 ──────────────────────────────────────────────────────────
def plot_per_stage_metrics(ax, cm, class_names):
    metrics_per = []
    for i in range(len(class_names)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp + 1e-8)
        rc = tp / (tp + fn + 1e-8)
        f1 = 2 * pr * rc / (pr + rc + 1e-8)
        metrics_per.append({"precision": pr, "recall": rc, "f1": f1})

    x     = np.arange(len(class_names))
    width = 0.25

    bars_p = ax.bar(x - width, [m["precision"] * 100 for m in metrics_per], width, label="Precision",  color=[c + "cc" for c in COLORS], edgecolor="white", linewidth=0.5)
    bars_r = ax.bar(x,         [m["recall"]    * 100 for m in metrics_per], width, label="Recall",     color=COLORS,                     edgecolor="white", linewidth=0.5)
    bars_f = ax.bar(x + width, [m["f1"]        * 100 for m in metrics_per], width, label="F1",         color=[c + "88" for c in COLORS], edgecolor="white", linewidth=0.5)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title("단계별 Precision / Recall / F1", fontsize=12, fontweight="bold", pad=8)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)


# ── 4. 추론 확률 분포 ──────────────────────────────────────────────────────────
def plot_demo_probs(ax, demo_results, class_names):
    n_stages = len(demo_results)
    x        = np.arange(len(class_names))
    width    = 0.18

    for i, res in enumerate(demo_results):
        offset = (i - n_stages / 2 + 0.5) * width
        bars   = ax.bar(x + offset, res["probs"] * 100, width,
                        color=COLORS[i], alpha=0.85,
                        label=f"실제: {class_names[res['stage']]}",
                        edgecolor="white", linewidth=0.5)
        # 정답 바에 테두리 강조
        bars[res["stage"]].set_edgecolor("black")
        bars[res["stage"]].set_linewidth(1.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"→{n}" for n in class_names], fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("확률 (%)")
    ax.set_xlabel("예측 클래스")
    ax.set_title("추론 데모 — 단계별 확률 분포", fontsize=12, fontweight="bold", pad=8)
    ax.legend(fontsize=8, framealpha=0.8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)


# ── 5. 단계 프로파일 레이더 ────────────────────────────────────────────────────
def plot_stage_radar(ax, profiles):
    labels  = ["보행\n주파수", "떨림\n강도", "비대칭", "팔 흔들림\n(역)", "보폭\n불규칙"]
    n_feat  = len(labels)
    angles  = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=6)
    ax.set_title("단계별 보행 특성 프로파일", fontsize=12, fontweight="bold", pad=20)

    for stage, p in profiles.items():
        # 정규화: 정상 대비 변화 정도
        freq_norm = p["freq"] / 1.80                        # 높을수록 정상
        trem_norm = min(p["tremor_amp"] / 0.22, 1.0)       # 높을수록 이상
        asym_norm = min(p["asymmetry"] / 0.30, 1.0)        # 높을수록 이상
        arm_inv   = 1 - p["arm_swing_scale"]               # 높을수록 감소
        irreg     = 1 - p["stride_regularity"]              # 높을수록 불규칙

        vals = [freq_norm, trem_norm, asym_norm, arm_inv, irreg]
        vals += vals[:1]

        ax.plot(angles, vals, color=COLORS[stage], linewidth=2, label=CLASS_NAMES[stage])
        ax.fill(angles, vals, color=COLORS[stage], alpha=0.12)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)


# ── 6. 단계 진행 열 지도 ───────────────────────────────────────────────────────
def plot_stage_progression(ax, profiles):
    features = {
        "보행 주파수 (Hz)": [p["freq"] for p in profiles.values()],
        "떨림 강도":        [p["tremor_amp"] for p in profiles.values()],
        "좌우 비대칭 (%)":  [p["asymmetry"] * 100 for p in profiles.values()],
        "팔 흔들림 (%)":    [p["arm_swing_scale"] * 100 for p in profiles.values()],
        "보폭 규칙성":      [p["stride_regularity"] for p in profiles.values()],
    }

    data = np.array(list(features.values()))
    # 행별 min-max 정규화 (시각화용)
    row_min = data.min(axis=1, keepdims=True)
    row_max = data.max(axis=1, keepdims=True)
    data_n  = (data - row_min) / (row_max - row_min + 1e-8)

    im = ax.imshow(data_n, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(list(features.keys()), fontsize=9)
    ax.set_title("단계별 생체 지표 진행", fontsize=12, fontweight="bold", pad=8)

    raw_vals = list(features.values())
    for i, row in enumerate(raw_vals):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if data_n[i, j] > 0.7 else "black", fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="정규화 값 (이상↑)")


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/prodromal/best_model.pt")
    parser.add_argument("--out",        default="outputs/prodromal/training_result.png")
    args = parser.parse_args()

    setup_korean_font()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 체크포인트 로드
    ckpt   = torch.load(args.checkpoint, weights_only=True)
    config = ckpt["config"]
    print(f"체크포인트 로드: 에폭 {ckpt['epoch']}, 최고 검증 정확도 {ckpt['val_accuracy']*100:.2f}%")

    # 모델 복원
    model = MultimodalGaitNet(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 데이터 재생성 (동일 시드 → 동일 분할)
    data_cfg = config["data"]
    raw = generate_prodromal_dataset(
        num_samples_per_stage=data_cfg.get("num_samples_per_stage", 100),
        num_frames=data_cfg["sequence_length"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )
    dataset = MultimodalGaitDataset(
        raw,
        sequence_length=data_cfg["sequence_length"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )
    total  = len(dataset)
    n_tr   = int(total * data_cfg["train_split"])
    n_val  = int(total * data_cfg["val_split"])
    n_test = total - n_tr - n_val
    _, _, test_ds = random_split(dataset, [n_tr, n_val, n_test],
                                 generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 테스트 평가
    criterion = nn.CrossEntropyLoss()
    all_probs, all_labels = get_all_probs_and_labels(model, test_loader, device)
    all_preds  = all_probs.argmax(axis=1)
    test_m     = compute_metrics(all_labels, all_preds)
    cm         = test_m["confusion_matrix"]
    print(f"테스트 정확도: {test_m['accuracy']*100:.2f}%  F1: {test_m['f1_macro']:.4f}")

    # 추론 데모
    demo = get_demo_probs(model, raw, config, device)

    # ── 그림 구성 (3×2 + 레이더) ─────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0f172a")

    ax_style = dict(facecolor="#1e293b")
    text_col = "#e2e8f0"

    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.35,
                          left=0.06, right=0.97, top=0.93, bottom=0.05)

    ax1 = fig.add_subplot(gs[0, 0], **ax_style)   # loss
    ax2 = fig.add_subplot(gs[0, 1], **ax_style)   # accuracy
    ax3 = fig.add_subplot(gs[0, 2], **ax_style)   # stage progression
    ax4 = fig.add_subplot(gs[1, 0], **ax_style)   # confusion matrix
    ax5 = fig.add_subplot(gs[1, 1], **ax_style)   # per-stage bar
    ax6 = fig.add_subplot(gs[1, 2], **ax_style)   # demo probs
    ax7 = fig.add_subplot(gs[2, :2], polar=True, facecolor="#1e293b")  # radar
    ax8 = fig.add_subplot(gs[2, 2], **ax_style)   # summary

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax8]:
        ax.tick_params(colors=text_col)
        ax.xaxis.label.set_color(text_col)
        ax.yaxis.label.set_color(text_col)
        ax.title.set_color(text_col)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    ax7.tick_params(colors=text_col)
    ax7.title.set_color(text_col)

    # 제목
    fig.suptitle(
        "질환 발병 전(전임상~초기징후) 이상징후 감지 — 학습 결과 시각화",
        fontsize=16, fontweight="bold", color=text_col, y=0.97,
    )

    # 서브플롯 그리기
    history = ckpt["history"]
    plot_training_curves(ax1, ax2, history)
    plot_stage_progression(ax3, STAGE_PROFILES)
    plot_confusion_matrix(ax4, cm, CLASS_NAMES)
    plot_per_stage_metrics(ax5, cm, CLASS_NAMES)
    plot_demo_probs(ax6, demo, CLASS_NAMES)
    plot_stage_radar(ax7, STAGE_PROFILES)

    # 다크 테마 적용
    for ax in [ax1, ax2, ax3, ax5, ax6]:
        ax.set_facecolor("#1e293b")
        ax.grid(True, color="#334155", alpha=0.5)
        for txt in ax.get_xticklabels() + ax.get_yticklabels():
            txt.set_color(text_col)
        ax.legend_ and ax.legend_.get_frame().set_facecolor("#1e293b") and [t.set_color(text_col) for t in ax.legend_.get_texts()]

    # 요약 텍스트 박스
    ax8.axis("off")
    acc     = test_m["accuracy"] * 100
    f1      = test_m["f1_macro"]
    summary_lines = [
        "[ 최종 성능 요약 ]",
        "",
        f"  테스트 정확도   {acc:.2f}%",
        f"  F1 (macro)     {f1:.4f}",
        f"  정밀도          {test_m['precision']:.4f}",
        f"  재현율          {test_m['recall']:.4f}",
        "",
        "[ 단계별 F1 ]",
        "",
    ]
    for i in range(4):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp + 1e-8); rc = tp / (tp + fn + 1e-8)
        f1s = 2 * pr * rc / (pr + rc + 1e-8)
        summary_lines.append(f"  {CLASS_NAMES[i]:5s}  F1 {f1s:.3f}  ({int(cm[i,i])}/{int(cm[i,:].sum())})")

    summary_lines += [
        "",
        "[ 데이터 ]",
        "",
        f"  학습 {n_tr} / 검증 {n_val} / 테스트 {n_test}",
        f"  에폭 {ckpt['epoch']} (최고 검증 acc)",
        "",
        "[ 모델 ]",
        "",
        "  MultimodalGaitNet",
        f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}",
    ]

    ax8.text(0.05, 0.97, "\n".join(summary_lines),
             transform=ax8.transAxes,
             fontsize=10, verticalalignment="top",
             fontfamily=FONT_KR, color=text_col,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#0f172a",
                       edgecolor="#334155", linewidth=1.5))

    # 저장
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n시각화 저장 완료: {out_path.resolve()}")


if __name__ == "__main__":
    main()
