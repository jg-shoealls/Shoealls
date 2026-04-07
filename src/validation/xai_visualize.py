"""Attention 시각화 및 설명 가능한 AI (XAI) 모듈.

제공 기능:
    1. Cross-modal attention weight 히트맵
    2. Grad-CAM 기반 압력 센서 중요 영역 표시
    3. 모달리티별 기여도 분석
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────────
MODALITY_NAMES = ["IMU", "Pressure", "Skeleton"]
MODALITY_NAMES_KR = ["IMU (관성센서)", "족저압 센서", "스켈레톤"]
CLASS_NAMES_KR = ["정상 보행", "절뚝거림", "운동실조", "파킨슨"]

MODALITY_COLORS = ["#2196F3", "#FF5722", "#009688"]
CMAP_ATTENTION = "YlOrRd"
CMAP_GRADCAM = "jet"

FOOT_ZONE_LABELS = {
    "toes":             {"rows": (0, 3),  "cols": (0, 8), "label": "발가락"},
    "forefoot_medial":  {"rows": (3, 7),  "cols": (0, 4), "label": "앞발(내)"},
    "forefoot_lateral": {"rows": (3, 7),  "cols": (4, 8), "label": "앞발(외)"},
    "midfoot_medial":   {"rows": (7, 11), "cols": (0, 4), "label": "중족(내)"},
    "midfoot_lateral":  {"rows": (7, 11), "cols": (4, 8), "label": "중족(외)"},
    "heel_medial":      {"rows": (11, 16), "cols": (0, 4), "label": "뒤꿈치(내)"},
    "heel_lateral":     {"rows": (11, 16), "cols": (4, 8), "label": "뒤꿈치(외)"},
}


# ═══════════════════════════════════════════════════════════════════════
# 1. Cross-Modal Attention Weight Heatmap
# ═══════════════════════════════════════════════════════════════════════

def plot_cross_modal_attention(
    cross_attn_weights: np.ndarray,
    save_path: Path | str,
    sample_idx: int = 0,
    title: str = "교차 모달 Attention 히트맵",
):
    """Cross-modal attention weight를 히트맵으로 시각화.

    Args:
        cross_attn_weights: (B, num_heads, 3, 3) attention weights
            from CrossModalEvidenceCollector.
        save_path: 저장 경로.
        sample_idx: 배치 내 샘플 인덱스.
        title: 그래프 제목.
    """
    # (num_heads, 3, 3)
    weights = cross_attn_weights[sample_idx]
    num_heads = weights.shape[0]

    fig, axes = plt.subplots(1, num_heads + 1, figsize=(4 * (num_heads + 1), 4))

    # Per-head heatmaps
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(weights[h], cmap=CMAP_ATTENTION, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(MODALITY_NAMES, fontsize=9)
        ax.set_yticklabels(MODALITY_NAMES, fontsize=9)
        ax.set_title(f"Head {h + 1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Key (참조 대상)", fontsize=9)
        if h == 0:
            ax.set_ylabel("Query (질의 모달리티)", fontsize=9)

        # Annotate
        for i in range(3):
            for j in range(3):
                val = weights[h, i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    # Average across heads
    ax = axes[num_heads]
    avg = weights.mean(axis=0)
    im = ax.imshow(avg, cmap=CMAP_ATTENTION, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(MODALITY_NAMES, fontsize=9)
    ax.set_yticklabels(MODALITY_NAMES, fontsize=9)
    ax.set_title("평균 (All Heads)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Key (참조 대상)", fontsize=9)
    for i in range(3):
        for j in range(3):
            val = avg[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label="Attention Weight")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# 2. Grad-CAM for Pressure Sensor
# ═══════════════════════════════════════════════════════════════════════

def compute_pressure_gradcam(
    engine,
    batch: dict,
    target_class: int | None = None,
    layer_name: str = "cnn",
) -> np.ndarray:
    """Grad-CAM을 사용하여 압력 센서 중요 영역을 계산.

    PressureEncoder의 마지막 CNN 레이어에 대해 Grad-CAM을 수행하여
    모델 판단에 중요한 압력 영역을 식별합니다.

    Args:
        engine: GaitReasoningEngine 인스턴스.
        batch: 입력 배치 딕셔너리 {imu, pressure, skeleton}.
        target_class: 타겟 클래스 인덱스. None이면 예측 클래스 사용.
        layer_name: 후킹할 CNN 레이어 이름.

    Returns:
        gradcam_map: (B, T, H, W) Grad-CAM 히트맵 (0~1 정규화).
    """
    engine.eval()

    # Enable gradients temporarily
    pressure_input = batch["pressure"].clone().requires_grad_(True)
    modified_batch = {
        "imu": batch["imu"],
        "pressure": pressure_input,
        "skeleton": batch["skeleton"],
    }

    # Hook into the last conv layer of PressureEncoder
    activations = {}
    gradients = {}

    target_layer = engine.pressure_encoder.cnn[-5]  # last Conv2d before final pool

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass (with grad enabled)
        imu_feat = engine.imu_encoder(modified_batch["imu"])
        pressure_feat = engine.pressure_encoder(modified_batch["pressure"])
        skeleton_feat = engine.skeleton_encoder(modified_batch["skeleton"])

        modality_features = [imu_feat, pressure_feat, skeleton_feat]

        # Run through anomaly detectors
        anomaly_results = []
        for feat, detector in zip(modality_features, engine.anomaly_detectors):
            anomaly_results.append(detector(feat))

        deviations = [r["deviation"] for r in anomaly_results]
        evidence = engine.evidence_collector(modality_features, deviations)

        anomaly_context = sum(deviations) / 3
        diagnosis = engine.diagnosis_chain(
            evidence["evidence_embedding"], anomaly_context,
        )

        logits = diagnosis["hypothesis_logits"]  # (B, C)

        if target_class is None:
            target_class = logits.argmax(dim=-1)  # (B,)

        # Backward pass for target class
        engine.zero_grad()
        if isinstance(target_class, int):
            score = logits[:, target_class].sum()
        else:
            score = logits.gather(1, target_class.unsqueeze(1)).sum()
        score.backward(retain_graph=False)

        # Compute Grad-CAM
        act = activations["value"]   # (B*T, C, H', W')
        grad = gradients["value"]    # (B*T, C, H', W')

        # Global average pooling of gradients -> channel weights
        weights = grad.mean(dim=(2, 3), keepdim=True)  # (B*T, C, 1, 1)

        # Weighted combination
        cam = (weights * act).sum(dim=1, keepdim=True)  # (B*T, 1, H', W')
        cam = F.relu(cam)  # Only positive contributions

        # Upsample to original pressure grid size (16, 8)
        B, T = batch["pressure"].shape[:2]
        H, W = batch["pressure"].shape[3], batch["pressure"].shape[4]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)  # (B*T, H, W)
        cam = cam.reshape(B, T, H, W)

        # Normalize per sample
        cam_np = cam.detach().cpu().numpy()
        for b in range(B):
            cmax = cam_np[b].max()
            if cmax > 1e-8:
                cam_np[b] /= cmax

        return cam_np

    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def plot_pressure_gradcam(
    gradcam_map: np.ndarray,
    pressure_data: np.ndarray,
    save_path: Path | str,
    sample_idx: int = 0,
    num_frames: int = 6,
    title: str = "Grad-CAM: 압력 센서 중요 영역",
):
    """Grad-CAM 히트맵을 압력 데이터 위에 오버레이하여 시각화.

    Args:
        gradcam_map: (B, T, H, W) Grad-CAM 결과.
        pressure_data: (B, T, 1, H, W) 원본 압력 데이터.
        save_path: 저장 경로.
        sample_idx: 배치 내 샘플 인덱스.
        num_frames: 표시할 프레임 수.
        title: 그래프 제목.
    """
    cam = gradcam_map[sample_idx]    # (T, H, W)
    press = pressure_data[sample_idx]  # (T, 1, H, W)
    if press.ndim == 4:
        press = press[:, 0]  # (T, H, W)

    T = cam.shape[0]
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)

    fig, axes = plt.subplots(3, num_frames, figsize=(3 * num_frames, 10))

    for col, t in enumerate(frame_indices):
        # Row 1: Raw pressure
        ax = axes[0, col]
        im0 = ax.imshow(press[t], cmap="Blues", aspect="auto", vmin=0, vmax=press.max())
        ax.set_title(f"t={t}", fontsize=9)
        if col == 0:
            ax.set_ylabel("압력 원본", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Row 2: Grad-CAM
        ax = axes[1, col]
        im1 = ax.imshow(cam[t], cmap=CMAP_GRADCAM, aspect="auto", vmin=0, vmax=1)
        if col == 0:
            ax.set_ylabel("Grad-CAM", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Row 3: Overlay
        ax = axes[2, col]
        ax.imshow(press[t], cmap="gray", aspect="auto", vmin=0, vmax=press.max())
        ax.imshow(cam[t], cmap=CMAP_GRADCAM, aspect="auto", alpha=0.6, vmin=0, vmax=1)
        if col == 0:
            ax.set_ylabel("오버레이", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Zone annotations on the last overlay
    ax = axes[2, -1]
    for zone_name, zone_def in FOOT_ZONE_LABELS.items():
        r0, r1 = zone_def["rows"]
        c0, c1 = zone_def["cols"]
        mid_r = (r0 + r1) / 2.0 - 0.5
        mid_c = (c0 + c1) / 2.0 - 0.5
        ax.text(mid_c, mid_r, zone_def["label"], ha="center", va="center",
                fontsize=6, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

    fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.02, pad=0.04,
                 label="Grad-CAM 중요도")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def compute_zone_importance(gradcam_map: np.ndarray, sample_idx: int = 0) -> dict:
    """Grad-CAM 맵에서 해부학적 영역별 중요도를 계산.

    Args:
        gradcam_map: (B, T, H, W) Grad-CAM 결과.
        sample_idx: 배치 내 샘플 인덱스.

    Returns:
        영역별 평균 Grad-CAM 값 딕셔너리.
    """
    cam = gradcam_map[sample_idx]  # (T, H, W)
    avg_cam = cam.mean(axis=0)     # (H, W) time-averaged

    zone_scores = {}
    for zone_name, zone_def in FOOT_ZONE_LABELS.items():
        r0, r1 = zone_def["rows"]
        c0, c1 = zone_def["cols"]
        zone_scores[zone_name] = {
            "importance": float(avg_cam[r0:r1, c0:c1].mean()),
            "peak": float(avg_cam[r0:r1, c0:c1].max()),
            "label": zone_def["label"],
        }

    return zone_scores


# ═══════════════════════════════════════════════════════════════════════
# 3. Modality Contribution Analysis
# ═══════════════════════════════════════════════════════════════════════

def plot_modality_contribution(
    modality_weights: np.ndarray,
    cross_support: np.ndarray,
    anomaly_results: list[dict],
    save_path: Path | str,
    sample_idx: int = 0,
    title: str = "모달리티별 기여도 분석",
):
    """각 센서 모달리티의 최종 판단 기여도를 종합 시각화.

    Args:
        modality_weights: (B, 3) 모달리티 기여 가중치.
        cross_support: (B, 3) 교차 모달 지지도.
        anomaly_results: 3개 모달리티의 anomaly detection 결과 리스트.
        save_path: 저장 경로.
        sample_idx: 배치 내 샘플 인덱스.
        title: 그래프 제목.
    """
    weights = modality_weights[sample_idx]   # (3,)
    support = cross_support[sample_idx]      # (3,)

    anomaly_names = [
        "좌우 비대칭", "리듬 불규칙", "진폭 이상", "주파수 이상",
        "공간 패턴 이상", "시간 지연", "떨림", "보행 동결",
    ]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # ── Panel 1: Modality contribution pie chart ──
    ax = fig.add_subplot(gs[0, 0])
    wedges, texts, autotexts = ax.pie(
        weights, labels=MODALITY_NAMES_KR, autopct="%1.1f%%",
        colors=MODALITY_COLORS, startangle=90,
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("모달리티 기여도", fontsize=12, fontweight="bold")

    # ── Panel 2: Contribution vs Cross-support bar chart ──
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(3)
    w = 0.35
    bars1 = ax.bar(x - w / 2, weights, w, label="기여도", color=MODALITY_COLORS, alpha=0.85)
    bars2 = ax.bar(x + w / 2, support, w, label="교차 지지도",
                   color=MODALITY_COLORS, alpha=0.4, edgecolor=MODALITY_COLORS, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(MODALITY_NAMES, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("기여도 vs 교차 지지도", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    # Value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── Panel 3: Temporal anomaly heatmaps (stacked) ──
    ax = fig.add_subplot(gs[0, 2])
    raw_heatmaps = [anom["temporal_heatmap"][sample_idx].cpu().numpy() for anom in anomaly_results]
    # Resample to uniform length (max T across modalities)
    max_t = max(h.shape[0] for h in raw_heatmaps)
    heatmaps = np.zeros((3, max_t))
    for idx, h in enumerate(raw_heatmaps):
        if h.shape[0] == max_t:
            heatmaps[idx] = h
        else:
            x_old = np.linspace(0, 1, h.shape[0])
            x_new = np.linspace(0, 1, max_t)
            heatmaps[idx] = np.interp(x_new, x_old, h)

    im = ax.imshow(heatmaps, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(MODALITY_NAMES, fontsize=10)
    ax.set_xlabel("시간 프레임", fontsize=10)
    ax.set_title("시간축 이상 히트맵", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="이상 점수")

    # ── Panel 4-6: Per-modality anomaly scores ──
    for m_idx in range(3):
        ax = fig.add_subplot(gs[1, m_idx])
        scores = anomaly_results[m_idx]["anomaly_scores"][sample_idx].cpu().numpy()

        colors_bar = ["#C0392B" if s > 0.5 else "#3498DB" for s in scores]
        bars = ax.barh(range(len(anomaly_names)), scores, color=colors_bar, alpha=0.85)
        ax.set_yticks(range(len(anomaly_names)))
        ax.set_yticklabels(anomaly_names, fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("이상 점수", fontsize=9)
        ax.set_title(f"{MODALITY_NAMES_KR[m_idx]}", fontsize=11, fontweight="bold",
                     color=MODALITY_COLORS[m_idx])
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="임계값")
        ax.grid(True, alpha=0.2, axis="x")
        ax.invert_yaxis()

        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{score:.2f}", va="center", fontsize=8)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# 4. XAI 종합 대시보드
# ═══════════════════════════════════════════════════════════════════════

def plot_xai_dashboard(
    result: dict,
    batch: dict,
    engine,
    save_path: Path | str,
    sample_idx: int = 0,
):
    """Attention 시각화 + Grad-CAM + 모달리티 기여도를 하나의 대시보드로 생성.

    Args:
        result: GaitReasoningEngine.reason() 출력.
        batch: 입력 배치 딕셔너리.
        engine: GaitReasoningEngine 인스턴스.
        save_path: 저장 경로.
        sample_idx: 배치 내 샘플 인덱스.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pred = result["prediction"][sample_idx].item()
    probs = result["calibrated_probs"][sample_idx].cpu().numpy()
    uncertainty = result["uncertainty"][sample_idx].item()

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35)

    # ── Title bar ──
    fig.suptitle(
        f"XAI 대시보드  |  판정: {CLASS_NAMES_KR[pred]} ({probs[pred]:.1%})  |  불확실성: {uncertainty:.1%}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # ── Row 0, Col 0-1: Cross-modal attention heatmap (avg heads) ──
    cross_attn = result["evidence"]["cross_attn_weights"][sample_idx].cpu().numpy()
    avg_attn = cross_attn.mean(axis=0)  # (3, 3)

    ax = fig.add_subplot(gs[0, 0:2])
    im = ax.imshow(avg_attn, cmap=CMAP_ATTENTION, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(MODALITY_NAMES, fontsize=10)
    ax.set_yticklabels(MODALITY_NAMES, fontsize=10)
    ax.set_xlabel("Key (참조 대상)", fontsize=10)
    ax.set_ylabel("Query (질의)", fontsize=10)
    ax.set_title("Cross-Modal Attention (평균)", fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            val = avg_attn[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 0, Col 2: Modality contribution pie ──
    ax = fig.add_subplot(gs[0, 2])
    weights = result["evidence"]["modality_weights"][sample_idx].cpu().numpy()
    wedges, texts, autotexts = ax.pie(
        weights, labels=MODALITY_NAMES_KR, autopct="%1.1f%%",
        colors=MODALITY_COLORS, startangle=90, textprops={"fontsize": 8},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("모달리티 기여도", fontsize=12, fontweight="bold")

    # ── Row 0, Col 3: Class probabilities ──
    ax = fig.add_subplot(gs[0, 3])
    class_colors = ["#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    bars = ax.barh(range(4), probs, color=class_colors, alpha=0.85)
    ax.set_yticks(range(4))
    ax.set_yticklabels(CLASS_NAMES_KR, fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("확률", fontsize=10)
    ax.set_title("진단 확률 분포", fontsize=12, fontweight="bold")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}", va="center", fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # ── Row 1: Grad-CAM pressure overlay (selected frames) ──
    try:
        gradcam = compute_pressure_gradcam(engine, batch, target_class=pred)
        cam = gradcam[sample_idx]   # (T, H, W)
        press = batch["pressure"][sample_idx].cpu().numpy()
        if press.ndim == 4:
            press = press[:, 0]  # (T, H, W)

        T = cam.shape[0]
        n_frames = 4
        frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

        for col, t in enumerate(frame_indices):
            ax = fig.add_subplot(gs[1, col])
            ax.imshow(press[t], cmap="gray", aspect="auto", vmin=0, vmax=press.max())
            ax.imshow(cam[t], cmap=CMAP_GRADCAM, aspect="auto", alpha=0.6, vmin=0, vmax=1)
            ax.set_title(f"Grad-CAM (t={t})", fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

            # Zone labels on first frame
            if col == 0:
                for zone_def in FOOT_ZONE_LABELS.values():
                    r0, r1 = zone_def["rows"]
                    c0, c1 = zone_def["cols"]
                    mid_r = (r0 + r1) / 2.0 - 0.5
                    mid_c = (c0 + c1) / 2.0 - 0.5
                    ax.text(mid_c, mid_r, zone_def["label"], ha="center", va="center",
                            fontsize=5, color="white", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4))

        # Zone importance summary
        zone_scores = compute_zone_importance(gradcam, sample_idx)
    except Exception:
        # If Grad-CAM fails (e.g. no grad support), show placeholder
        for col in range(4):
            ax = fig.add_subplot(gs[1, col])
            ax.text(0.5, 0.5, "Grad-CAM\n(사용 불가)", ha="center", va="center",
                    fontsize=12, color="gray", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        zone_scores = None

    # ── Row 2: Per-modality anomaly scores + temporal heatmap ──
    anomaly_names = [
        "비대칭", "리듬", "진폭", "주파수",
        "공간", "지연", "떨림", "동결",
    ]

    for m_idx in range(3):
        ax = fig.add_subplot(gs[2, m_idx])
        scores = result["anomaly_results"][m_idx]["anomaly_scores"][sample_idx].cpu().numpy()
        colors_bar = ["#C0392B" if s > 0.5 else "#95A5A6" for s in scores]
        ax.barh(range(8), scores, color=colors_bar, alpha=0.85)
        ax.set_yticks(range(8))
        ax.set_yticklabels(anomaly_names, fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_title(MODALITY_NAMES_KR[m_idx], fontsize=10, fontweight="bold",
                     color=MODALITY_COLORS[m_idx])
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.4)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis="x")

    # Zone importance bar (if available)
    ax = fig.add_subplot(gs[2, 3])
    if zone_scores is not None:
        zone_names = [v["label"] for v in zone_scores.values()]
        zone_vals = [v["importance"] for v in zone_scores.values()]
        zone_colors = [CMAP_GRADCAM] * len(zone_names)
        cmap = plt.cm.get_cmap("YlOrRd")
        zone_colors = [cmap(v) for v in zone_vals]
        ax.barh(range(len(zone_names)), zone_vals, color=zone_colors, alpha=0.85)
        ax.set_yticks(range(len(zone_names)))
        ax.set_yticklabels(zone_names, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Grad-CAM 중요도", fontsize=9)
        ax.set_title("발 영역별 중요도", fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis="x")
    else:
        ax.text(0.5, 0.5, "영역별 분석\n(사용 불가)", ha="center", va="center",
                fontsize=12, color="gray", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved XAI dashboard: {save_path}")
