"""HFMultimodalGaitNet 파인튜닝 전략 유틸.

3가지 전략:
  strategy1_feature_extraction — 백본 전체 동결, proj+fusion+classifier만 학습
  strategy2_partial            — 마지막 N개 Transformer 레이어만 학습
  strategy3_lora               — LoRA 어댑터 삽입 (peft 필요)

모든 함수는 model을 in-place로 수정하고 (학습 가능 파라미터 수)를 반환한다.
"""

from __future__ import annotations
import logging
from typing import Optional
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False

def _unfreeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True

def _get_patchtst_encoder_layers(backbone) -> list:
    """PatchTSTModel → encoder layer 목록"""
    return list(backbone.encoder.layers)

def _get_videomae_encoder_layers(backbone) -> list:
    """VideoMAEModel → encoder layer 목록"""
    return list(backbone.encoder.layer)


# ── Strategy 1: Feature Extraction ───────────────────────────────────────────

def strategy1_feature_extraction(model: nn.Module) -> dict:
    """백본 전체를 동결하고 projection + fusion + classifier만 학습.

    학습 파라미터: proj(IMU) + PressureEncoder + proj(Skeleton) + fusion + classifier
    동결 파라미터: PatchTSTModel(전체) + VideoMAEModel(전체)
    """
    # 전체 동결
    _freeze_all(model)

    # 투영 레이어 열기
    _unfreeze_all(model.imu_encoder.proj)
    _unfreeze_all(model.skeleton_encoder.proj)
    _unfreeze_all(model.skeleton_encoder.dropout)

    # Pressure 인코더 전체 열기 (HF 백본 없음)
    _unfreeze_all(model.pressure_encoder)

    # Fusion + Classifier 열기
    _unfreeze_all(model.fusion)
    _unfreeze_all(model.classifier)

    n = _count_trainable(model)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"[Strategy1] 학습 파라미터: {n:,} / {total:,} ({n/total*100:.1f}%)")
    return {"trainable": n, "total": total, "strategy": "feature_extraction"}


# ── Strategy 2: Partial Fine-tuning ──────────────────────────────────────────

def strategy2_partial(model: nn.Module, unfreeze_last_n: int = 1) -> dict:
    """초기 레이어 동결, 마지막 N개 Transformer 블록 + head 학습.

    Args:
        unfreeze_last_n: 열 마지막 인코더 레이어 수 (기본 1)
    """
    # 전체 동결
    _freeze_all(model)

    # IMU PatchTST: 마지막 N 레이어 열기
    imu_backbone = model.imu_encoder.backbone
    if not model.imu_encoder._FallbackConv1DEncoder if False else True:
        try:
            ptst_layers = _get_patchtst_encoder_layers(imu_backbone)
            for layer in ptst_layers[-unfreeze_last_n:]:
                _unfreeze_all(layer)
            logger.info(f"  PatchTST: 마지막 {unfreeze_last_n}/{len(ptst_layers)} 레이어 열림")
        except Exception as e:
            logger.warning(f"  PatchTST 레이어 접근 실패: {e}")

    # Skeleton VideoMAE: 마지막 N 레이어 열기
    skel_backbone = model.skeleton_encoder.backbone
    if not getattr(model.skeleton_encoder, "_use_fallback", True):
        try:
            vmae_layers = _get_videomae_encoder_layers(skel_backbone)
            for layer in vmae_layers[-unfreeze_last_n:]:
                _unfreeze_all(layer)
            logger.info(f"  VideoMAE: 마지막 {unfreeze_last_n}/{len(vmae_layers)} 레이어 열림")
        except Exception as e:
            logger.warning(f"  VideoMAE 레이어 접근 실패: {e}")

    # Projection + Pressure + Fusion + Classifier 열기
    _unfreeze_all(model.imu_encoder.proj)
    _unfreeze_all(model.skeleton_encoder.proj)
    _unfreeze_all(model.skeleton_encoder.dropout)
    _unfreeze_all(model.pressure_encoder)
    _unfreeze_all(model.fusion)
    _unfreeze_all(model.classifier)

    n = _count_trainable(model)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"[Strategy2] 학습 파라미터: {n:,} / {total:,} ({n/total*100:.1f}%)")
    return {"trainable": n, "total": total, "strategy": "partial"}


# ── Strategy 3: LoRA ──────────────────────────────────────────────────────────

def strategy3_lora(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> dict:
    """LoRA 어댑터를 Attention Q/V 레이어에 삽입한다.

    peft 라이브러리가 없으면 수동 LoRA로 fallback.

    Args:
        r: LoRA rank (낮을수록 파라미터 절약, 8~16이 일반적)
        lora_alpha: 스케일 계수 (보통 r의 2배)
        lora_dropout: LoRA 드롭아웃
    """
    # 전체 동결
    _freeze_all(model)

    # Pressure + Fusion + Classifier 열기 (HF 백본 아님)
    _unfreeze_all(model.imu_encoder.proj)
    _unfreeze_all(model.skeleton_encoder.proj)
    _unfreeze_all(model.skeleton_encoder.dropout)
    _unfreeze_all(model.pressure_encoder)
    _unfreeze_all(model.fusion)
    _unfreeze_all(model.classifier)

    # LoRA 삽입 시도
    lora_applied = {"patchtst": False, "videomae": False}

    # PatchTST LoRA
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        ptst_backbone = model.imu_encoder.backbone
        lora_cfg_ptst = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model.imu_encoder.backbone = get_peft_model(ptst_backbone, lora_cfg_ptst)
        lora_applied["patchtst"] = True
        logger.info(f"  PatchTST LoRA 적용 (r={r}, target: q_proj, v_proj)")
    except Exception as e:
        logger.warning(f"  PatchTST peft LoRA 실패 ({e}), 수동 LoRA 사용")
        _apply_manual_lora(model.imu_encoder.backbone, ["q_proj", "v_proj"], r, lora_alpha, lora_dropout)
        lora_applied["patchtst"] = "manual"

    # VideoMAE LoRA
    if not getattr(model.skeleton_encoder, "_use_fallback", True):
        try:
            from peft import LoraConfig, get_peft_model
            vmae_backbone = model.skeleton_encoder.backbone
            lora_cfg_vmae = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "value"],
                bias="none",
            )
            model.skeleton_encoder.backbone = get_peft_model(vmae_backbone, lora_cfg_vmae)
            lora_applied["videomae"] = True
            logger.info(f"  VideoMAE LoRA 적용 (r={r}, target: query, value)")
        except Exception as e:
            logger.warning(f"  VideoMAE peft LoRA 실패 ({e}), 수동 LoRA 사용")
            _apply_manual_lora(model.skeleton_encoder.backbone, ["query", "value"], r, lora_alpha, lora_dropout)
            lora_applied["videomae"] = "manual"

    n = _count_trainable(model)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"[Strategy3] 학습 파라미터: {n:,} / {total:,} ({n/total*100:.1f}%)")
    return {"trainable": n, "total": total, "strategy": "lora", "lora_applied": lora_applied}


# ── 수동 LoRA (peft 없는 환경 fallback) ──────────────────────────────────────

class _LoRALinear(nn.Module):
    """단일 Linear에 LoRA 어댑터를 추가한다.

    y = W·x + (B·A·x) * (alpha/r)
    W: frozen, A/B: trainable
    """
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        d_in, d_out = linear.in_features, linear.out_features
        self.linear = linear  # frozen
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)
        self.scale  = alpha / r
        self.drop   = nn.Dropout(dropout)

        # 초기화: A는 정규분포, B는 0 (처음엔 LoRA 출력 = 0)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(self.drop(x))) * self.scale


def _apply_manual_lora(
    module: nn.Module,
    target_names: list[str],
    r: int,
    alpha: float,
    dropout: float,
) -> None:
    """모듈 트리에서 target_names 와 일치하는 Linear를 LoRALinear로 교체."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, _LoRALinear(child, r, alpha, dropout))
            # LoRA weight는 자동으로 requires_grad=True
        else:
            _apply_manual_lora(child, target_names, r, alpha, dropout)


# ── 레이어별 학습률 옵티마이저 ────────────────────────────────────────────────

def make_layerwise_optimizer(
    model: nn.Module,
    base_lr: float = 3e-4,
    backbone_lr_ratio: float = 0.1,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """백본은 낮은 LR, 나머지는 base_lr 사용하는 AdamW 옵티마이저."""
    import torch
    backbone_params, head_params = [], []
    backbone_keywords = {"imu_encoder.backbone", "skeleton_encoder.backbone"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = any(kw in name for kw in backbone_keywords)
        (backbone_params if is_backbone else head_params).append(param)

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": base_lr * backbone_lr_ratio},
        {"params": head_params,     "lr": base_lr},
    ], weight_decay=weight_decay)


import torch  # 하단 import (순환 방지)
