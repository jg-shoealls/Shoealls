"""Multi-task multimodal gait analysis network.

Extends MultimodalGaitNet with multiple task heads:
  - Gait pattern classification (original)
  - Disease classification
  - Fall risk prediction
  - Gait phase detection

Architecture:
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ IMU Data │   │ Pressure │   │ Skeleton │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
    ┌────▼─────┐   ┌────▼─────┐   ┌────▼─────┐
    │1D-CNN    │   │2D-CNN    │   │ST-GCN    │
    │+BiLSTM   │   │          │   │          │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────┬───────┴──────┬───────┘
                │              │
    ┌───────────▼──────┐  ┌────▼──────────────┐
    │ Cross-Modal      │  │ Temporal Features  │
    │ Attention Fusion │  │ (pre-fusion seq)   │
    └───────┬──────────┘  └─────┬─────────────┘
            │                   │
    ┌───────▼────────┐          │
    │ Fused Embedding │         │
    │ (B, 128)        │         │
    └──┬──┬──┬────────┘         │
       │  │  │                  │
  ┌────▼┐ │ ┌▼────────┐  ┌─────▼──────────┐
  │Gait │ │ │Disease   │  │Fall Risk       │
  │Cls  │ │ │Cls +     │  │(Global+Temporal)│
  │     │ │ │Severity  │  │                │
  └─────┘ │ └──────────┘  └────────────────┘
          │
    ┌─────▼────────┐
    │Gait Phase    │
    │Detection     │
    │(per-frame)   │
    └──────────────┘
"""

import torch
import torch.nn as nn

from .encoders import IMUEncoder, PressureEncoder, SkeletonEncoder
from .fusion import CrossModalAttentionFusion
from .task_heads import (
    DiseaseClassificationHead,
    FallRiskPredictionHead,
    GaitPhaseDetectionHead,
)


class MultitaskGaitNet(nn.Module):
    """Multi-task multimodal gait analysis network."""

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        data_cfg = config["data"]
        task_cfg = config.get("tasks", {})

        embed_dim = model_cfg["fusion"]["embed_dim"]

        # ── Shared encoders (same as base model) ──────────────────────
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

        # ── Shared fusion ─────────────────────────────────────────────
        fusion_cfg = model_cfg["fusion"]
        self.fusion = CrossModalAttentionFusion(
            embed_dim=embed_dim,
            num_heads=fusion_cfg["num_heads"],
            ff_dim=fusion_cfg["ff_dim"],
            num_layers=fusion_cfg["num_layers"],
            num_modalities=3,
            dropout=fusion_cfg["dropout"],
        )

        # ── Task 1: Gait pattern classification (original) ───────────
        cls_cfg = model_cfg["classifier"]
        gait_layers = []
        in_dim = embed_dim
        for h_dim in cls_cfg["hidden_dims"]:
            gait_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cls_cfg["dropout"]),
            ])
            in_dim = h_dim
        gait_layers.append(nn.Linear(in_dim, data_cfg["num_classes"]))
        self.gait_classifier = nn.Sequential(*gait_layers)

        # ── Task 2: Disease classification ────────────────────────────
        disease_cfg = task_cfg.get("disease", {})
        self.disease_head = DiseaseClassificationHead(
            embed_dim=embed_dim,
            num_diseases=disease_cfg.get("num_diseases", 7),
            hidden_dims=disease_cfg.get("hidden_dims", [256, 128]),
            dropout=disease_cfg.get("dropout", 0.4),
        )

        # ── Task 3: Fall risk prediction ──────────────────────────────
        fall_cfg = task_cfg.get("fall_risk", {})
        self.fall_risk_head = FallRiskPredictionHead(
            embed_dim=embed_dim,
            temporal_hidden=fall_cfg.get("temporal_hidden", 64),
            num_temporal_layers=fall_cfg.get("num_temporal_layers", 1),
            dropout=fall_cfg.get("dropout", 0.3),
        )

        # ── Task 4: Gait phase detection ─────────────────────────────
        phase_cfg = task_cfg.get("gait_phase", {})
        self.gait_phase_head = GaitPhaseDetectionHead(
            embed_dim=embed_dim,
            num_phases=phase_cfg.get("num_phases", 8),
            dropout=phase_cfg.get("dropout", 0.3),
        )

        # ── Task selection ────────────────────────────────────────────
        self.active_tasks = task_cfg.get("active", [
            "gait", "disease", "fall_risk", "gait_phase",
        ])

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: Dict with 'imu', 'pressure', 'skeleton'.

        Returns:
            Dict with outputs per active task.
        """
        # Encode each modality
        imu_features = self.imu_encoder(batch["imu"])           # (B, T1, D)
        pressure_features = self.pressure_encoder(batch["pressure"])  # (B, T2, D)
        skeleton_features = self.skeleton_encoder(batch["skeleton"])  # (B, T3, D)

        modality_features = [imu_features, pressure_features, skeleton_features]

        # Fused global embedding
        fused = self.fusion(modality_features)  # (B, D)

        # Temporal features (concatenated pre-fusion sequences)
        temporal_cat = torch.cat(modality_features, dim=1)  # (B, T1+T2+T3, D)

        # ── Compute active task outputs ──
        outputs = {}

        if "gait" in self.active_tasks:
            outputs["gait_logits"] = self.gait_classifier(fused)

        if "disease" in self.active_tasks:
            disease_out = self.disease_head(fused)
            outputs["disease_logits"] = disease_out["disease_logits"]
            outputs["severity"] = disease_out["severity"]

        if "fall_risk" in self.active_tasks:
            fall_out = self.fall_risk_head(fused, temporal_cat)
            outputs["risk_logits"] = fall_out["risk_logits"]
            outputs["risk_score"] = fall_out["risk_score"]
            outputs["time_to_fall"] = fall_out["time_to_fall"]

        if "gait_phase" in self.active_tasks:
            # Use skeleton temporal features for phase detection
            phase_out = self.gait_phase_head(skeleton_features)
            outputs["phase_logits"] = phase_out["phase_logits"]

        return outputs

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_task_param_breakdown(self) -> dict:
        """Get parameter count per component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "shared_encoders": {
                "imu_encoder": _count(self.imu_encoder),
                "pressure_encoder": _count(self.pressure_encoder),
                "skeleton_encoder": _count(self.skeleton_encoder),
            },
            "shared_fusion": _count(self.fusion),
            "task_heads": {
                "gait_classifier": _count(self.gait_classifier),
                "disease_head": _count(self.disease_head),
                "fall_risk_head": _count(self.fall_risk_head),
                "gait_phase_head": _count(self.gait_phase_head),
            },
            "total": self.get_num_params(),
        }
