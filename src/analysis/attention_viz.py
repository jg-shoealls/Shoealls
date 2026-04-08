"""Attention 시각화 및 설명 가능한 AI (XAI) 모듈.

Cross-modal attention weight 추출, Grad-CAM, 모달리티 기여도 분석을 통해
모델의 판단 근거를 시각적으로 설명합니다.

Attention Visualization and Explainable AI (XAI) module.
Extracts cross-modal attention weights, Grad-CAM, and modality contribution
analysis to visually explain model decisions.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class AttentionExtractor:
    """크로스모달 퓨전 모듈에서 어텐션 가중치를 추출.

    Extracts attention weights from cross-modal fusion module.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._attention_weights = {}
        self._hooks = []

    def _register_hooks(self):
        """퓨전 모듈의 MultiheadAttention에 hook 등록."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                hook = module.register_forward_hook(self._attention_hook(name))
                self._hooks.append(hook)

    def _attention_hook(self, name: str):
        def hook_fn(module, input, output):
            # MultiheadAttention returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                weights = output[1]
                if weights is not None:
                    self._attention_weights[name] = weights.detach().cpu()
        return hook_fn

    def extract(self, batch: dict) -> dict:
        """배치 데이터에 대한 어텐션 가중치 추출.

        Args:
            batch: 모델 입력 딕셔너리.

        Returns:
            attention_weights: {layer_name: (B, num_heads, T_q, T_k)} 형태의 딕셔너리.
        """
        self._attention_weights = {}
        self._register_hooks()
        try:
            self.model.eval()
            with torch.no_grad():
                self.model(batch)
        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks = []
        return dict(self._attention_weights)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class GradCAM:
    """Grad-CAM: 압력 인코더의 공간적 중요 영역 시각화.

    Grad-CAM for pressure encoder spatial region importance visualization.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "pressure_encoder"):
        self.model = model
        self.target_layer_name = target_layer_name
        self._gradients = None
        self._activations = None
        self._hooks = []

    def _find_target_layer(self) -> Optional[torch.nn.Module]:
        """대상 레이어 찾기."""
        for name, module in self.model.named_modules():
            if self.target_layer_name in name and isinstance(module, torch.nn.Conv2d):
                return module
        # Conv2d를 찾지 못하면 마지막 conv 레이어 반환
        last_conv = None
        for name, module in self.model.named_modules():
            if self.target_layer_name in name:
                for subname, submodule in module.named_modules():
                    if isinstance(submodule, torch.nn.Conv2d):
                        last_conv = submodule
        return last_conv

    def _register_hooks(self, target_layer: torch.nn.Module):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def compute(self, batch: dict, target_class: Optional[int] = None) -> np.ndarray:
        """Grad-CAM 히트맵 계산.

        Args:
            batch: 모델 입력 딕셔너리.
            target_class: 대상 클래스. None이면 예측 클래스 사용.

        Returns:
            heatmap: (B, H, W) Grad-CAM 히트맵.
        """
        target_layer = self._find_target_layer()
        if target_layer is None:
            raise ValueError(f"Target layer '{self.target_layer_name}' not found")

        self._register_hooks(target_layer)
        self.model.eval()

        # Enable gradients temporarily
        for p in self.model.parameters():
            p.requires_grad_(True)

        try:
            logits = self.model(batch)
            if target_class is None:
                target_class = logits.argmax(dim=-1)

            # Backward for target class
            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            for i in range(logits.size(0)):
                cls = target_class[i] if isinstance(target_class, torch.Tensor) else target_class
                one_hot[i, cls] = 1.0
            logits.backward(gradient=one_hot, retain_graph=True)

            if self._gradients is None or self._activations is None:
                return np.zeros((logits.size(0), 16, 8))

            # Global average pooling of gradients
            weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            cam = (weights * self._activations).sum(dim=1)  # (B, H, W)
            cam = F.relu(cam)

            # Normalize
            B = cam.size(0)
            cam_flat = cam.view(B, -1)
            cam_min = cam_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
            cam_max = cam_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            return cam.cpu().numpy()

        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks = []
            for p in self.model.parameters():
                p.requires_grad_(False)


class ModalityContributionAnalyzer:
    """모달리티별 기여도 분석.

    제로 마스킹 기반으로 각 모달리티가 예측에 얼마나 기여하는지 분석.
    Analyzes per-modality contribution using zero-masking approach.
    """

    MODALITY_NAMES = {"imu": "IMU (관성센서)", "pressure": "족저압 센서", "skeleton": "스켈레톤"}

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @torch.no_grad()
    def analyze(self, batch: dict) -> dict:
        """각 모달리티의 기여도를 분석.

        Args:
            batch: 모델 입력 딕셔너리 (label 포함).

        Returns:
            contributions: {modality: importance_score} (높을수록 중요)
            drop_accuracies: {modality: accuracy_when_dropped}
        """
        self.model.eval()
        labels = batch.get("label")
        input_batch = {k: v for k, v in batch.items() if k != "label"}

        # Full model prediction
        full_logits = self.model(input_batch)
        full_probs = F.softmax(full_logits, dim=-1)
        full_preds = full_logits.argmax(dim=-1)

        contributions = {}
        drop_results = {}

        for modality in ["imu", "pressure", "skeleton"]:
            if modality not in input_batch:
                continue

            # Zero-mask this modality
            masked_batch = dict(input_batch)
            masked_batch[modality] = torch.zeros_like(input_batch[modality])
            masked_logits = self.model(masked_batch)
            masked_probs = F.softmax(masked_logits, dim=-1)

            # KL divergence: how much prediction changes
            kl_div = F.kl_div(
                masked_probs.log(), full_probs, reduction="batchmean"
            ).item()
            contributions[modality] = kl_div

            # Accuracy drop
            if labels is not None:
                masked_preds = masked_logits.argmax(dim=-1)
                masked_acc = (masked_preds == labels).float().mean().item()
                full_acc = (full_preds == labels).float().mean().item()
                drop_results[modality] = {
                    "accuracy_without": masked_acc,
                    "accuracy_drop": full_acc - masked_acc,
                }

        # Normalize contributions to sum to 1
        total = sum(contributions.values()) + 1e-8
        normalized = {k: v / total for k, v in contributions.items()}

        return {
            "contributions": normalized,
            "raw_kl_divergence": contributions,
            "drop_results": drop_results,
        }


def plot_attention_heatmap(
    attention_weights: dict,
    save_path: Optional[str] = None,
    sample_idx: int = 0,
):
    """어텐션 가중치 히트맵 시각화.

    Args:
        attention_weights: AttentionExtractor.extract() 출력.
        save_path: 저장 경로.
        sample_idx: 시각화할 샘플 인덱스.
    """
    num_layers = len(attention_weights)
    if num_layers == 0:
        print("No attention weights to visualize")
        return

    fig, axes = plt.subplots(1, min(num_layers, 4), figsize=(5 * min(num_layers, 4), 4))
    if num_layers == 1:
        axes = [axes]

    for idx, (name, weights) in enumerate(attention_weights.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        w = weights[sample_idx].mean(dim=0).numpy()  # Average over heads
        im = ax.imshow(w, aspect="auto", cmap="viridis")
        ax.set_title(name.split(".")[-2] if "." in name else name, fontsize=9)
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Cross-Modal Attention Weights", fontsize=12)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gradcam_heatmap(
    heatmap: np.ndarray,
    save_path: Optional[str] = None,
    sample_idx: int = 0,
):
    """Grad-CAM 히트맵 시각화 (족저압 공간).

    Args:
        heatmap: GradCAM.compute() 출력 (B, H, W).
        save_path: 저장 경로.
        sample_idx: 시각화할 샘플 인덱스.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    cam = heatmap[sample_idx]
    im = ax.imshow(cam, cmap="jet", aspect="auto", origin="lower")
    ax.set_title("Grad-CAM: Pressure Sensor Importance", fontsize=11)
    ax.set_xlabel("Width")
    ax.set_ylabel("Height (Heel -> Toe)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_modality_contributions(
    analysis_result: dict,
    save_path: Optional[str] = None,
):
    """모달리티 기여도 막대 차트.

    Args:
        analysis_result: ModalityContributionAnalyzer.analyze() 출력.
        save_path: 저장 경로.
    """
    contributions = analysis_result["contributions"]
    names_kr = ModalityContributionAnalyzer.MODALITY_NAMES

    labels = [names_kr.get(k, k) for k in contributions.keys()]
    values = list(contributions.values())
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Contribution bar chart
    ax = axes[0]
    bars = ax.bar(labels, values, color=colors[:len(labels)])
    ax.set_ylabel("Contribution (normalized)")
    ax.set_title("Modality Contribution")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", fontsize=10)

    # Accuracy drop chart
    drop_results = analysis_result.get("drop_results", {})
    if drop_results:
        ax2 = axes[1]
        drop_labels = [names_kr.get(k, k) for k in drop_results.keys()]
        drop_values = [v["accuracy_drop"] for v in drop_results.values()]
        bars2 = ax2.bar(drop_labels, drop_values, color=colors[:len(drop_labels)])
        ax2.set_ylabel("Accuracy Drop")
        ax2.set_title("Accuracy Drop When Modality Removed")
        for bar, val in zip(bars2, drop_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "No labels provided", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_title("Accuracy Drop (N/A)")

    plt.suptitle("Modality Importance Analysis", fontsize=13)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
