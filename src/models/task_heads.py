"""Multi-task heads for disease classification and fall risk prediction.

Extends the base MultimodalGaitNet with:
  1. Disease classification head (neurological/musculoskeletal/cardiopulmonary)
  2. Fall risk prediction head (binary risk + continuous risk score)
  3. Gait phase detection head (stance/swing/double-support)
"""

import torch
import torch.nn as nn


class DiseaseClassificationHead(nn.Module):
    """질환 분류 헤드.

    융합 임베딩(128-dim)으로부터 근골격계/신경계/심폐계 질환을 분류.

    Target classes:
        0: 정상
        1: 뇌졸중 (Stroke)
        2: 파킨슨병 (Parkinson's Disease)
        3: 다발성 경화증 (Multiple Sclerosis)
        4: 골관절염 (Osteoarthritis)
        5: 만성 요통 (Chronic Low Back Pain)
        6: 말초신경병증 (Peripheral Neuropathy)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_diseases: int = 7,
        hidden_dims: list = None,
        dropout: float = 0.4,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        layers = []
        in_dim = embed_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_diseases)

        # Severity estimation (regression: 0.0 ~ 1.0)
        self.severity_head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused_embedding: torch.Tensor) -> dict:
        """
        Args:
            fused_embedding: (B, embed_dim) from fusion layer.

        Returns:
            Dict with 'disease_logits' (B, num_diseases) and 'severity' (B, 1).
        """
        features = self.feature_extractor(fused_embedding)
        return {
            "disease_logits": self.classifier(features),
            "severity": self.severity_head(features),
        }


class FallRiskPredictionHead(nn.Module):
    """낙상 위험도 예측 헤드.

    두 가지 출력:
      1. 이진 분류: 고위험 / 저위험
      2. 연속 위험 점수: 0.0 (안전) ~ 1.0 (위험)

    시간적 패턴(불안정성 누적)을 반영하기 위해
    융합 전 시퀀스 특징도 함께 사용.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        temporal_hidden: int = 64,
        num_temporal_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Branch A: 융합 임베딩 기반 (전역 특징)
        self.global_branch = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Branch B: 시계열 기반 (시간적 불안정성 패턴)
        self.temporal_encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=temporal_hidden,
            num_layers=num_temporal_layers,
            batch_first=True,
            dropout=dropout if num_temporal_layers > 1 else 0,
        )

        combined_dim = 64 + temporal_hidden

        # 이진 분류 (고위험/저위험)
        self.risk_classifier = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

        # 연속 위험 점수
        self.risk_score = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 낙상 시점 예측 (몇 보행 주기 이내에 낙상 가능성이 있는지)
        self.time_to_fall = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Softplus(),  # 양수 출력 (보행 주기 수)
        )

    def forward(
        self,
        fused_embedding: torch.Tensor,
        temporal_features: torch.Tensor,
    ) -> dict:
        """
        Args:
            fused_embedding: (B, embed_dim) from fusion layer.
            temporal_features: (B, T, embed_dim) pre-fusion sequence features.

        Returns:
            Dict with 'risk_logits', 'risk_score', 'time_to_fall'.
        """
        # Global branch
        global_feat = self.global_branch(fused_embedding)  # (B, 64)

        # Temporal branch
        _, h_n = self.temporal_encoder(temporal_features)  # h_n: (1, B, 64)
        temporal_feat = h_n[-1]  # (B, 64)

        combined = torch.cat([global_feat, temporal_feat], dim=1)  # (B, 128)

        return {
            "risk_logits": self.risk_classifier(combined),   # (B, 2)
            "risk_score": self.risk_score(combined),          # (B, 1)
            "time_to_fall": self.time_to_fall(combined),      # (B, 1)
        }


class GaitPhaseDetectionHead(nn.Module):
    """보행 위상 검출 헤드 (프레임 단위).

    각 시간 프레임에 대해 보행 위상을 분류:
        0: 초기 접지기 (Initial Contact)
        1: 하중 반응기 (Loading Response)
        2: 중간 입각기 (Mid Stance)
        3: 말기 입각기 (Terminal Stance)
        4: 전유각기 (Pre-Swing)
        5: 초기 유각기 (Initial Swing)
        6: 중간 유각기 (Mid Swing)
        7: 말기 유각기 (Terminal Swing)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_phases: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.temporal_refine = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv1d(embed_dim // 2, num_phases, kernel_size=1)

    def forward(self, temporal_features: torch.Tensor) -> dict:
        """
        Args:
            temporal_features: (B, T, embed_dim) sequence features.

        Returns:
            Dict with 'phase_logits' (B, num_phases, T).
        """
        # (B, T, D) -> (B, D, T) for Conv1d
        x = temporal_features.permute(0, 2, 1)
        x = self.temporal_refine(x)
        phase_logits = self.classifier(x)  # (B, num_phases, T)

        return {"phase_logits": phase_logits}
