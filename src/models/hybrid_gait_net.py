"""HuggingFace 사전학습 인코더 기반 하이브리드 보행 분석 모델.

기존 MultimodalGaitNet의 커스텀 인코더를 경량 사전학습 모델로 교체합니다:
  IMU     → LIMUBERTEncoder   (~62K)
  Pressure→ MobileNetV2Encoder (~1.7M)
  Skeleton→ CTRGCNEncoder     (~1.5M)
  Fusion  → CrossModalAttentionFusion (기존 유지)

기존 MultimodalGaitNet과 동일한 인터페이스 (forward, get_num_params 등).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .fusion import CrossModalAttentionFusion
from .pretrained_encoders import (
    LIMUBERTEncoder,
    MobileNetV2PressureEncoder,
    CTRGCNEncoder,
)


class HybridGaitNet(nn.Module):
    """사전학습 인코더 + CrossModal 퓨전 하이브리드 모델.

    사용법:
        model = HybridGaitNet.from_config(config)
        logits = model({"imu": ..., "pressure": ..., "skeleton": ...})

    파인튜닝 전략:
        1단계: model.freeze_encoders() → 분류기만 학습
        2단계: model.unfreeze_fusion() → 퓨전 포함 학습
        3단계: model.unfreeze_all() → 전체 미세 조정
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 128,
        # IMU encoder (LIMU-BERT)
        imu_embed_dim: int = 72,
        imu_heads: int = 4,
        imu_layers: int = 4,
        imu_patch_size: int = 8,
        imu_dropout: float = 0.1,
        # Pressure encoder (MobileNetV2)
        pressure_pretrained: bool = True,
        pressure_width_mult: float = 0.35,
        pressure_dropout: float = 0.1,
        # Skeleton encoder (CTR-GCN)
        skeleton_joints: int = 17,
        skeleton_gcn_channels: Optional[list] = None,
        skeleton_dropout: float = 0.2,
        # Fusion
        fusion_heads: int = 4,
        fusion_layers: int = 2,
        fusion_ff_dim: int = 256,
        fusion_dropout: float = 0.1,
        # Classifier head
        classifier_hidden: Optional[list] = None,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()

        self.imu_encoder = LIMUBERTEncoder(
            in_channels=6,
            embed_dim=imu_embed_dim,
            num_heads=imu_heads,
            num_layers=imu_layers,
            patch_size=imu_patch_size,
            dropout=imu_dropout,
            embed_dim_out=embed_dim,
        )

        self.pressure_encoder = MobileNetV2PressureEncoder(
            embed_dim=embed_dim,
            dropout=pressure_dropout,
            pretrained=pressure_pretrained,
            width_mult=pressure_width_mult,
        )

        self.skeleton_encoder = CTRGCNEncoder(
            in_channels=3,
            num_joints=skeleton_joints,
            gcn_channels=skeleton_gcn_channels or [64, 128],
            embed_dim=embed_dim,
            dropout=skeleton_dropout,
        )

        self.fusion = CrossModalAttentionFusion(
            embed_dim=embed_dim,
            num_heads=fusion_heads,
            ff_dim=fusion_ff_dim,
            num_layers=fusion_layers,
            num_modalities=3,
            dropout=fusion_dropout,
        )

        hidden = classifier_hidden or [128, 64]
        layers = []
        in_dim = embed_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True), nn.Dropout(classifier_dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, config: dict, pretrained: bool = True) -> "HybridGaitNet":
        """설정 딕셔너리에서 모델 생성.

        기존 MultimodalGaitNet.from_config()와 동일한 config 형식을 지원합니다.
        """
        data_cfg = config["data"]
        model_cfg = config.get("model", {})
        fusion_cfg = model_cfg.get("fusion", {})
        cls_cfg = model_cfg.get("classifier", {})

        embed_dim = fusion_cfg.get("embed_dim", 128)

        return cls(
            num_classes=data_cfg["num_classes"],
            embed_dim=embed_dim,
            skeleton_joints=data_cfg.get("skeleton_joints", 17),
            fusion_heads=fusion_cfg.get("num_heads", 4),
            fusion_layers=fusion_cfg.get("num_layers", 2),
            fusion_ff_dim=fusion_cfg.get("ff_dim", 256),
            fusion_dropout=fusion_cfg.get("dropout", 0.1),
            classifier_hidden=cls_cfg.get("hidden_dims", [128, 64]),
            classifier_dropout=cls_cfg.get("dropout", 0.3),
            pressure_pretrained=pretrained,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: {"imu": (B,6,T), "pressure": (B,T,1,H,W), "skeleton": (B,3,T,J)}
        Returns:
            logits: (B, num_classes)
        """
        imu_feat = self.imu_encoder(batch["imu"])
        pressure_feat = self.pressure_encoder(batch["pressure"])
        skeleton_feat = self.skeleton_encoder(batch["skeleton"])

        fused = self.fusion([imu_feat, pressure_feat, skeleton_feat])
        return self.classifier(fused)

    # ── 파인튜닝 헬퍼 ─────────────────────────────────────────────

    def freeze_all(self):
        """모든 파라미터 동결."""
        for p in self.parameters():
            p.requires_grad = False

    def freeze_encoders(self):
        """Phase 1: 인코더 동결, 퓨전·분류기만 학습."""
        for enc in (self.imu_encoder, self.pressure_encoder, self.skeleton_encoder):
            for p in enc.parameters():
                p.requires_grad = False

    def unfreeze_fusion(self):
        """Phase 2: 퓨전 레이어 해동."""
        for p in self.fusion.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        """Phase 3: 전체 모델 해동."""
        for p in self.parameters():
            p.requires_grad = True

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_summary(self) -> str:
        lines = ["[HybridGaitNet 파라미터 현황]"]
        modules = {
            "IMU (LIMU-BERT)": self.imu_encoder,
            "Pressure (MobileNetV2)": self.pressure_encoder,
            "Skeleton (CTR-GCN)": self.skeleton_encoder,
            "Fusion": self.fusion,
            "Classifier": self.classifier,
        }
        total = 0
        for name, mod in modules.items():
            n = sum(p.numel() for p in mod.parameters())
            trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            lines.append(f"  {name:<24}: {n:>8,} params  ({trainable:>8,} trainable)")
            total += n
        lines.append(f"  {'합계':<24}: {total:>8,} params")
        return "\n".join(lines)
