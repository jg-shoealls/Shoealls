"""IMU + Pressure multimodal network for PD vs HC classification."""

import torch
import torch.nn as nn

from .encoders import IMUEncoder, PressureEncoder


class IMUPressureGaitNet(nn.Module):
    """Dual-modal gait classification network (IMU + plantar pressure).

    Architecture:
        IMU data      -> IMUEncoder (1D-CNN + BiLSTM)  -> mean-pool -> (B, embed_dim)
        Pressure data -> PressureEncoder (2D-CNN)       -> mean-pool -> (B, embed_dim)
                                                                         |
                                              concat -> MLP classifier -> logits
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        data_cfg = config["data"]

        embed_dim = model_cfg["fusion"]["embed_dim"]

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

        # Classifier: [imu_embed, pressure_embed] concatenated -> MLP
        cls_cfg = model_cfg["classifier"]
        layers = []
        in_dim = embed_dim * 2
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
        """
        Args:
            batch: dict with keys:
                'imu':      (B, 12, T)
                'pressure': (B, T, 1, 4, 8)

        Returns:
            logits: (B, num_classes)
        """
        # IMU: (B, 12, T) -> (B, T', embed_dim) -> mean -> (B, embed_dim)
        imu_feat = self.imu_encoder(batch["imu"]).mean(dim=1)

        # Pressure: (B, T, 1, H, W) -> (B, T, embed_dim) -> mean -> (B, embed_dim)
        pres_feat = self.pressure_encoder(batch["pressure"]).mean(dim=1)

        # Concatenate and classify
        fused = torch.cat([imu_feat, pres_feat], dim=1)  # (B, embed_dim * 2)
        return self.classifier(fused)
