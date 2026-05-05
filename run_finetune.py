"""파인튜닝 전략 비교 스크립트.

4가지 전략을 순차 실행하고 결과를 비교한다:
  0. Baseline     — 전체 학습 (동결 없음)
  1. Feature Ext  — 백본 전체 동결, head만 학습
  2. Partial      — 마지막 1개 레이어 + head 학습
  3. LoRA         — LoRA 어댑터 + head 학습
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# 1. VideoMAE 경량 패치 (CPU 학습용)
# ─────────────────────────────────────────────────────────────────────────────

VIDEOMAE_ARCH_PATCH = dict(
    image_size=64,
    patch_size=16,
    num_frames=4,
    tubelet_size=2,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=256,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
)

HF_CONFIG_OVERRIDE = dict(
    patchtst=dict(pretrained=False, d_model=128, num_layers=2),
    videomae=dict(pretrained=False, img_size=64, num_frames=4),
)


def _patch_videomae_build() -> None:
    """VideoMAESkeletonEncoder._build_backbone을 경량 설정으로 교체."""
    from src.models.hf_encoders import VideoMAESkeletonEncoder

    @staticmethod  # type: ignore[misc]
    def _lightweight_build(pretrained: bool, img_size: int, num_frames: int):
        try:
            from transformers import VideoMAEConfig, VideoMAEModel

            cfg = VideoMAEConfig(**VIDEOMAE_ARCH_PATCH)
            return VideoMAEModel(cfg), cfg.hidden_size, False
        except ImportError:
            from src.models.hf_encoders import _FallbackLSTMEncoder
            fb = _FallbackLSTMEncoder(3 * 17, 128)
            return fb, 128, True

    VideoMAESkeletonEncoder._build_backbone = _lightweight_build  # type: ignore[method-assign]


_patch_videomae_build()

# ─────────────────────────────────────────────────────────────────────────────
# 2. 설정 로드
# ─────────────────────────────────────────────────────────────────────────────

with open("configs/default.yaml") as f:
    BASE_CONFIG: dict = yaml.safe_load(f)

# HF 인코더 활성화 + 경량 설정 적용
BASE_CONFIG.setdefault("hf_encoders", {})
BASE_CONFIG["hf_encoders"]["enabled"] = True
for key, vals in HF_CONFIG_OVERRIDE.items():
    BASE_CONFIG["hf_encoders"].setdefault(key, {}).update(vals)

NUM_CLASSES = BASE_CONFIG.get("model", {}).get("num_classes", 4)
DEVICE = torch.device("cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 합성 데이터 생성
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(n_per_class: int = 40, batch_size: int = 8, seed: int = 42):
    rng = np.random.RandomState(seed)

    def _class_data(cls_idx):
        T, H, W, J = 128, 16, 8, 17  # T=128 required by PatchTST context_length
        # pressure: (B, T, 1, H, W) — PressureEncoder expects this shape
        pressure = rng.randn(n_per_class, T, 1, H, W).astype(np.float32) * 0.3
        pressure[:, :, 0, cls_idx * 3 % H: (cls_idx * 3 % H) + 4, :] += 1.0
        imu = rng.randn(n_per_class, 6, T).astype(np.float32)       # (B, 6, T)
        imu[:, cls_idx % 6, :] += 2.0
        skeleton = rng.randn(n_per_class, 3, T, J).astype(np.float32) * 0.2  # (B, 3, T, J)
        skeleton[:, :, :, cls_idx % J] += 0.5
        labels = np.full(n_per_class, cls_idx, dtype=np.int64)
        return pressure, imu, skeleton, labels

    parts = [_class_data(c) for c in range(NUM_CLASSES)]
    pressure = np.concatenate([p[0] for p in parts])
    imu      = np.concatenate([p[1] for p in parts])
    skeleton = np.concatenate([p[2] for p in parts])
    labels   = np.concatenate([p[3] for p in parts])

    idx = rng.permutation(len(labels))
    pressure, imu, skeleton, labels = pressure[idx], imu[idx], skeleton[idx], labels[idx]

    split = int(len(labels) * 0.8)
    def _ds(sl):
        return TensorDataset(
            torch.tensor(pressure[sl]),
            torch.tensor(imu[sl]),
            torch.tensor(skeleton[sl]),
            torch.tensor(labels[sl]),
        )

    train_loader = DataLoader(_ds(slice(None, split)), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(_ds(slice(split, None)),  batch_size=batch_size)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# 4. 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def _batch_to_dict(batch) -> dict:
    pressure, imu, skeleton, labels = batch
    return {
        "pressure": pressure.to(DEVICE),
        "imu":      imu.to(DEVICE),
        "skeleton": skeleton.to(DEVICE),
    }, labels.to(DEVICE)


def train_strategy(
    strategy_name: str,
    apply_fn,           # callable(model) → info_dict  or None for baseline
    epochs: int = 25,
    base_lr: float = 3e-4,
    use_layerwise: bool = False,
) -> dict:
    """단일 전략 학습 실행."""
    from src.models.hf_gait_net import HFMultimodalGaitNet
    from src.training.finetune_utils import make_layerwise_optimizer

    print(f"\n{'─'*60}")
    print(f"  전략: {strategy_name}")
    print(f"{'─'*60}")

    train_loader, val_loader = make_loaders()
    model = HFMultimodalGaitNet(BASE_CONFIG).to(DEVICE)

    # 전략 적용
    if apply_fn is not None:
        info = apply_fn(model)
        trainable = info["trainable"]
        total_params = info["total"]
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        info = {"strategy": "baseline"}

    print(f"  학습 파라미터: {trainable:,} / {total_params:,} ({trainable/total_params*100:.1f}%)")

    if use_layerwise and apply_fn is not None:
        optimizer = make_layerwise_optimizer(model, base_lr=base_lr)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr, weight_decay=1e-4,
        )

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── 학습 ──
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x_dict, labels = _batch_to_dict(batch)
            optimizer.zero_grad()
            logits = model(x_dict)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # ── 검증 ──
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_dict, labels = _batch_to_dict(batch)
                preds = model(x_dict).argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        # macro F1
        f1 = _macro_f1(all_labels, all_preds, NUM_CLASSES)
        avg_loss = total_loss / len(train_loader)

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)

        if acc > best_acc:
            best_acc = acc

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:2d}/{epochs} | loss={avg_loss:.4f} | val_acc={acc:.3f} | F1={f1:.3f} | {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  ✓ 완료 — best_acc={best_acc:.3f}  총 {elapsed:.1f}s ({elapsed/epochs:.2f}s/epoch)")

    return {
        "strategy": strategy_name,
        "history": history,
        "best_acc": best_acc,
        "best_f1": max(history["val_f1"]),
        "trainable_params": trainable,
        "total_params": total_params,
        "trainable_ratio": trainable / total_params,
        "elapsed_sec": elapsed,
        "sec_per_epoch": elapsed / epochs,
        "info": info,
    }


def _macro_f1(y_true: list, y_pred: list, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(f1s))


# ─────────────────────────────────────────────────────────────────────────────
# 5. 시각화
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results: list[dict], save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    names     = [r["strategy"] for r in results]
    colors    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    epochs    = range(1, len(results[0]["history"]["val_acc"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Fine-tuning Strategy Comparison", fontsize=14, fontweight="bold")

    # (1) Val Accuracy curves
    ax = axes[0, 0]
    for r, c in zip(results, colors):
        ax.plot(epochs, r["history"]["val_acc"], label=r["strategy"], color=c, lw=1.8)
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

    # (2) Train Loss curves
    ax = axes[0, 1]
    for r, c in zip(results, colors):
        ax.plot(epochs, r["history"]["train_loss"], label=r["strategy"], color=c, lw=1.8)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (3) Val F1 curves
    ax = axes[0, 2]
    for r, c in zip(results, colors):
        ax.plot(epochs, r["history"]["val_f1"], label=r["strategy"], color=c, lw=1.8)
    ax.set_title("Validation Macro-F1")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

    # (4) Best Accuracy bar
    ax = axes[1, 0]
    bars = ax.bar(names, [r["best_acc"] for r in results], color=colors, width=0.5)
    ax.set_title("Best Validation Accuracy")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{r['best_acc']:.3f}", ha="center", va="bottom", fontsize=9)

    # (5) Trainable params ratio bar
    ax = axes[1, 1]
    ratios = [r["trainable_ratio"] * 100 for r in results]
    bars = ax.bar(names, ratios, color=colors, width=0.5)
    ax.set_title("Trainable Parameters (%)")
    ax.set_ylabel("% of total params"); ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, pct in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    # (6) Summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["Strategy", "Best Acc", "Best F1", "Params%", "sec/ep"],
    ]
    for r in results:
        table_data.append([
            r["strategy"],
            f"{r['best_acc']:.3f}",
            f"{r['best_f1']:.3f}",
            f"{r['trainable_ratio']*100:.1f}%",
            f"{r['sec_per_epoch']:.2f}s",
        ])

    tbl = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        bbox=[0, 0.1, 1, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f1f2")
    ax.set_title("Summary", fontsize=11, pad=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  시각화 저장: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from src.training.finetune_utils import (
        strategy1_feature_extraction,
        strategy2_partial,
        strategy3_lora,
    )

    output_dir = Path("outputs/finetune")
    output_dir.mkdir(parents=True, exist_ok=True)

    EPOCHS = 25

    strategies = [
        ("Baseline",     None),
        ("Feature Ext",  strategy1_feature_extraction),
        ("Partial (N=1)", lambda m: strategy2_partial(m, unfreeze_last_n=1)),
        ("LoRA (r=8)",   lambda m: strategy3_lora(m, r=8, lora_alpha=16, lora_dropout=0.05)),
    ]

    results = []
    for name, fn in strategies:
        r = train_strategy(name, fn, epochs=EPOCHS, use_layerwise=(fn is not None))
        results.append(r)

    # ── 요약 출력 ──
    print("\n" + "=" * 60)
    print("  파인튜닝 전략 비교 요약")
    print("=" * 60)
    print(f"{'전략':<18} {'Best Acc':>9} {'Best F1':>9} {'학습 파라미터':>15} {'비율':>7} {'sec/ep':>8}")
    print("─" * 72)
    for r in results:
        print(
            f"{r['strategy']:<18} "
            f"{r['best_acc']:>9.3f} "
            f"{r['best_f1']:>9.3f} "
            f"{r['trainable_params']:>15,} "
            f"{r['trainable_ratio']*100:>6.1f}% "
            f"{r['sec_per_epoch']:>7.2f}s"
        )

    # ── 시각화 ──
    plot_comparison(results, output_dir / "finetune_comparison.png")

    print("\n  완료! outputs/finetune/ 확인")


if __name__ == "__main__":
    main()
