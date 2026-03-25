"""Main multimodal gait analysis network."""

import torch
import torch.nn as nn

from .encoders import IMUEncoder, PressureEncoder, SkeletonEncoder
from .fusion import CrossModalAttentionFusion


class MultimodalGaitNet(nn.Module):
    """End-to-end multimodal gait classification network.

    Architecture:
        IMU data      -> IMUEncoder (1D-CNN + BiLSTM)     -> features
        Pressure data -> PressureEncoder (2D-CNN)          -> features
        Skeleton data -> SkeletonEncoder (ST-GCN)          -> features
                                                              |
                              Cross-Modal Attention Fusion <--+
                                        |
                                   Classifier -> class predictions
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        data_cfg = config["data"]

        embed_dim = model_cfg["fusion"]["embed_dim"]

        # Modality encoders
        imu_cfg = model_cfg["imu_encoder"]
        self.imu_encoder = IMUEncoder(
            in_channels=data_cfg["imu_channels"],
            conv_channels=imu_cfg["conv_channels"],
            kernel_size=imu_cfg["kernel_size"],
            lstm_hidden=embed_dim,
            lstm_layers=imu_cfg["lstm_layers"],
            dropout=imu_cfg["dropout"],
        )

        pressure_cfg = model_cfg["pressure_encoder"]
        self.pressure_encoder = PressureEncoder(
            in_channels=1,
            conv_channels=pressure_cfg["conv_channels"],
            kernel_size=pressure_cfg["kernel_size"],
            embed_dim=embed_dim,
            dropout=pressure_cfg["dropout"],
        )

        skeleton_cfg = model_cfg["skeleton_encoder"]
        self.skeleton_encoder = SkeletonEncoder(
            in_channels=data_cfg["skeleton_dims"],
            num_joints=data_cfg["skeleton_joints"],
            gcn_channels=skeleton_cfg["gcn_channels"],
            temporal_kernel=skeleton_cfg["temporal_kernel"],
            embed_dim=embed_dim,
            dropout=skeleton_cfg["dropout"],
        )

        # Fusion module
        fusion_cfg = model_cfg["fusion"]
        self.fusion = CrossModalAttentionFusion(
            embed_dim=embed_dim,
            num_heads=fusion_cfg["num_heads"],
            ff_dim=fusion_cfg["ff_dim"],
            num_layers=fusion_cfg["num_layers"],
            num_modalities=3,
            dropout=fusion_cfg["dropout"],
        )

        # Classifier head
        cls_cfg = model_cfg["classifier"]
        classifier_layers = []
        in_dim = embed_dim
        for h_dim in cls_cfg["hidden_dims"]:
            classifier_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cls_cfg["dropout"]),
            ])
            in_dim = h_dim
        classifier_layers.append(nn.Linear(in_dim, data_cfg["num_classes"]))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: Dictionary with keys 'imu', 'pressure', 'skeleton'.

        Returns:
            Class logits of shape (B, num_classes).
        """
        imu_features = self.imu_encoder(batch["imu"])
        pressure_features = self.pressure_encoder(batch["pressure"])
        skeleton_features = self.skeleton_encoder(batch["skeleton"])

        fused = self.fusion([imu_features, pressure_features, skeleton_features])
        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
