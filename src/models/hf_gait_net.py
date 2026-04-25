"""HuggingFace 인코더 기반 멀티모달 보행 분석 네트워크.

MultimodalGaitNet 과 동일한 forward 인터페이스를 유지하면서
IMU → PatchTST, Skeleton → VideoMAE 로 인코더를 교체한다.
Pressure 인코더는 16×8 그리드에 특화된 기존 PressureEncoder 를 유지.

설정 예시 (configs/default.yaml 의 hf_encoders 블록):

  hf_encoders:
    enabled: true
    patchtst:
      pretrained: true
      patch_len: 16
      stride: 8
      d_model: 128
      num_heads: 4
      num_layers: 3
      dropout: 0.2
    videomae:
      pretrained: true
      img_size: 224
      num_frames: 16
      dropout: 0.2
"""

import torch
import torch.nn as nn

from .encoders import PressureEncoder
from .fusion import CrossModalAttentionFusion
from .hf_encoders import PatchTSTIMUEncoder, VideoMAESkeletonEncoder


class HFMultimodalGaitNet(nn.Module):
    """PatchTST(IMU) + PressureEncoder(압력) + VideoMAE(스켈레톤) 융합 네트워크.

    Architecture:
        IMU data      -> PatchTSTIMUEncoder        -> (B, T_p, D)
        Pressure data -> PressureEncoder (2D-CNN)  -> (B, T,   D)
        Skeleton data -> VideoMAESkeletonEncoder   -> (B, T_v, D)
                                                       |
                          Cross-Modal Attention Fusion <+
                                    |
                               Classifier -> class logits
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        data_cfg = config["data"]
        hf_cfg = config.get("hf_encoders", {})

        embed_dim = model_cfg["fusion"]["embed_dim"]

        # ── PatchTST IMU 인코더 ─────────────────────────────────────────
        ptst_cfg = hf_cfg.get("patchtst", {})
        self.imu_encoder = PatchTSTIMUEncoder(
            embed_dim=embed_dim,
            patch_len=ptst_cfg.get("patch_len", 16),
            stride=ptst_cfg.get("stride", 8),
            d_model=ptst_cfg.get("d_model", 128),
            num_heads=ptst_cfg.get("num_heads", 4),
            num_layers=ptst_cfg.get("num_layers", 3),
            dropout=ptst_cfg.get("dropout", 0.2),
            pretrained=ptst_cfg.get("pretrained", True),
            in_channels=data_cfg["imu_channels"],
        )

        # ── 기존 Pressure 인코더 (유지) ─────────────────────────────────
        pressure_cfg = model_cfg["pressure_encoder"]
        self.pressure_encoder = PressureEncoder(
            in_channels=1,
            conv_channels=pressure_cfg["conv_channels"],
            kernel_size=pressure_cfg["kernel_size"],
            embed_dim=embed_dim,
            dropout=pressure_cfg["dropout"],
        )

        # ── VideoMAE 스켈레톤 인코더 ────────────────────────────────────
        vmae_cfg = hf_cfg.get("videomae", {})
        self.skeleton_encoder = VideoMAESkeletonEncoder(
            embed_dim=embed_dim,
            img_size=vmae_cfg.get("img_size", 224),
            num_frames=vmae_cfg.get("num_frames", 16),
            pretrained=vmae_cfg.get("pretrained", True),
            dropout=vmae_cfg.get("dropout", 0.2),
        )

        # ── Cross-Modal Attention Fusion ────────────────────────────────
        fusion_cfg = model_cfg["fusion"]
        self.fusion = CrossModalAttentionFusion(
            embed_dim=embed_dim,
            num_heads=fusion_cfg["num_heads"],
            ff_dim=fusion_cfg["ff_dim"],
            num_layers=fusion_cfg["num_layers"],
            num_modalities=3,
            dropout=fusion_cfg["dropout"],
        )

        # ── 분류기 헤드 ─────────────────────────────────────────────────
        cls_cfg = model_cfg["classifier"]
        layers = []
        in_dim = embed_dim
        for h_dim in cls_cfg["hidden_dims"]:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cls_cfg["dropout"]),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, data_cfg["num_classes"]))
        self.classifier = nn.Sequential(*layers)

    def forward(self, batch: dict) -> torch.Tensor:
        """MultimodalGaitNet 과 동일한 인터페이스.

        Args:
            batch: {'imu': (B,6,T), 'pressure': (B,T,1,H,W), 'skeleton': (B,3,T,J)}
        Returns:
            logits (B, num_classes)
        """
        imu_feat      = self.imu_encoder(batch["imu"])
        pressure_feat = self.pressure_encoder(batch["pressure"])
        skeleton_feat = self.skeleton_encoder(batch["skeleton"])

        fused = self.fusion([imu_feat, pressure_feat, skeleton_feat])
        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
