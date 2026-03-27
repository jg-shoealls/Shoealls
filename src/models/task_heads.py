"""
src/models/task_heads.py
퇴행성 뇌질환 조기 감지 및 기타 태스크를 위한 네트워크 헤드 모듈
"""

import torch
import torch.nn as nn

class DiseaseClassificationHead(nn.Module):
    """
    미세한 보행 패턴 변화를 바탕으로 퇴행성 뇌질환(치매, 파킨슨, 알츠하이머 등)을 
    조기 감지하고 위험도를 평가하는 멀티태스크 헤드입니다.
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_diseases: int = 3,  # 예: 0: 정상, 1: 알츠하이머, 2: 파킨슨, 3: 치매 등
        hidden_dims: list = [256, 128], 
        dropout: float = 0.4
    ):
        super().__init__()
        
        # 특징 추출을 위한 공통 은닉층 구성
        layers = []
        in_dim = embed_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),     # 학습 안정화를 위한 배치 정규화
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # 1. 질병 종류 분류기 (Disease Classification)
        # 각 질병 클래스에 대한 확률값을 출력하기 위한 레이어
        self.disease_classifier = nn.Linear(in_dim, num_diseases)
        
        # 2. 발병 위험도/중증도 예측기 (Severity / Early Risk Score)
        # 0 ~ 1 사이의 연속적인 위험도 점수를 출력하여 5~10년 전 조기 징후를 수치화
        self.severity_predictor = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid() # 0 ~ 1 사이의 값으로 변환
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: 융합된 보행 특징 벡터 (B, embed_dim)
               B는 배치 크기(Batch Size)를 의미합니다.
               
        Returns:
            질병 분류 결과와 위험도 점수를 담은 딕셔너리
        """
        # 공통 특징 추출
        features = self.feature_extractor(x)
        
        # 질병 분류 로짓 (Logits)
        disease_logits = self.disease_classifier(features)
        
        # 조기 발병 위험도 점수 (0.0 ~ 1.0)
        severity_score = self.severity_predictor(features)
        
        return {
            "disease_logits": disease_logits,
            "severity": severity_score
        }


class FallRiskPredictionHead(nn.Module):
    """낙상 위험도 예측 헤드: 전역 + 시계열 GRU 기반 이중 분기."""

    def __init__(self, embed_dim: int = 128, temporal_hidden: int = 64,
                 temporal_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.global_branch = nn.Sequential(
            nn.Linear(embed_dim, temporal_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.temporal_branch = nn.GRU(
            input_size=embed_dim, hidden_size=temporal_hidden,
            num_layers=temporal_layers, batch_first=True,
        )
        combined_dim = temporal_hidden * 2
        self.risk_classifier = nn.Sequential(
            nn.Linear(combined_dim, 32), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )
        self.risk_score = nn.Sequential(
            nn.Linear(combined_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.time_to_fall = nn.Sequential(
            nn.Linear(combined_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1), nn.Softplus(),
        )

    def forward(self, fused: torch.Tensor, temporal: torch.Tensor) -> dict:
        g = self.global_branch(fused)
        _, h = self.temporal_branch(temporal)
        t = h[-1]
        combined = torch.cat([g, t], dim=-1)
        return {
            "risk_logits": self.risk_classifier(combined),
            "risk_score": self.risk_score(combined),
            "time_to_fall": self.time_to_fall(combined),
        }


class GaitPhaseDetectionHead(nn.Module):
    """보행 위상 분류 헤드: 프레임 단위 8-phase 분류."""

    def __init__(self, embed_dim: int = 128, num_phases: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        mid_dim = embed_dim // 2
        self.conv_refine = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(embed_dim, mid_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_dim), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv1d(mid_dim, num_phases, kernel_size=1)

    def forward(self, temporal: torch.Tensor) -> dict:
        x = temporal.permute(0, 2, 1)  # (B, D, T)
        x = self.conv_refine(x)
        phase_logits = self.classifier(x)  # (B, num_phases, T)
        return {"phase_logits": phase_logits}
