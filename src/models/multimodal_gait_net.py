"""신발 전용 멀티모달 보행 분석 네트워크.

신발에서만 추출 가능한 3개 모달리티:
    IMU (발목)     -> IMUEncoder (1D-CNN + BiLSTM)   -> 발 키네마틱스
    족저압 (인솔)  -> PressureEncoder (2D-CNN)        -> 압력 분포
    지자기+기압    -> MagBaroEncoder (1D-CNN + LSTM)  -> FOG/발지상고
                                                          |
                          Cross-Modal Attention Fusion <--+
                                    |
                               Classifier -> 4단계 분류
"""

import torch
import torch.nn as nn

from .encoders import IMUEncoder, PressureEncoder, MagBaroEncoder
from .fusion import CrossModalAttentionFusion


class MultimodalGaitNet(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        data_cfg  = config["data"]
        embed_dim = model_cfg["fusion"]["embed_dim"]

        # IMU 인코더 (발목 6축: 가속도 3 + 자이로 3)
        imu_cfg = model_cfg["imu_encoder"]
        self.imu_encoder = IMUEncoder(
            in_channels=data_cfg["imu_channels"],
            conv_channels=imu_cfg["conv_channels"],
            kernel_size=imu_cfg["kernel_size"],
            lstm_hidden=embed_dim,
            lstm_layers=imu_cfg["lstm_layers"],
            dropout=imu_cfg["dropout"],
        )

        # 족저압 인코더 (인솔 16×8 그리드)
        pres_cfg = model_cfg["pressure_encoder"]
        self.pressure_encoder = PressureEncoder(
            in_channels=1,
            conv_channels=pres_cfg["conv_channels"],
            kernel_size=pres_cfg["kernel_size"],
            embed_dim=embed_dim,
            dropout=pres_cfg["dropout"],
        )

        # 지자기+기압 인코더 (mx,my,mz,heading,altitude — 5채널)
        mb_cfg = model_cfg["mag_baro_encoder"]
        self.mag_baro_encoder = MagBaroEncoder(
            in_channels=data_cfg.get("mag_baro_channels", 5),
            conv_channels=mb_cfg["conv_channels"],
            kernel_size=mb_cfg["kernel_size"],
            lstm_hidden=embed_dim,
            lstm_layers=mb_cfg["lstm_layers"],
            dropout=mb_cfg["dropout"],
        )

        # 크로스 모달 어텐션 융합 (3 모달리티)
        fusion_cfg = model_cfg["fusion"]
        self.fusion = CrossModalAttentionFusion(
            embed_dim=embed_dim,
            num_heads=fusion_cfg["num_heads"],
            ff_dim=fusion_cfg["ff_dim"],
            num_layers=fusion_cfg["num_layers"],
            num_modalities=3,
            dropout=fusion_cfg["dropout"],
        )

        # 분류기
        cls_cfg = model_cfg["classifier"]
        layers, in_dim = [], embed_dim
        for h in cls_cfg["hidden_dims"]:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True), nn.Dropout(cls_cfg["dropout"])]
            in_dim = h
        layers.append(nn.Linear(in_dim, data_cfg["num_classes"]))
        self.classifier = nn.Sequential(*layers)

    def forward(self, batch: dict) -> torch.Tensor:
        imu_feat  = self.imu_encoder(batch["imu"])
        pres_feat = self.pressure_encoder(batch["pressure"])
        mb_feat   = self.mag_baro_encoder(batch["mag_baro"])

        fused = self.fusion([imu_feat, pres_feat, mb_feat])
        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
