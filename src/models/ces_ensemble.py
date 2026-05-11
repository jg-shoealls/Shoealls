"""Shoealls AI Ensemble Model.

LSTM Autoencoder와 Transformer의 출력을 결합하고 
임상 규칙 기반 특징을 통합하여 최종 이상 점수를 산출합니다.
"""

import torch
import torch.nn as nn
from src.models.gait_lstm import GaitLSTM
from src.models.gait_transformer import GaitTransformer, GaitTransformerConfig

class ShoeallsEnsembleModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_classes: int = 3,
                 window_size: int = 30):
        super(ShoeallsEnsembleModel, self).__init__()

        # 1. LSTM (단기 패턴 및 이상 감지)
        self.lstm_model = GaitLSTM(input_dim, hidden_dim, num_layers=2, num_classes=num_classes)

        # 2. Transformer (장기 추세 감지) — gait_transformer.GaitTransformer 사용
        self.transformer_model = GaitTransformer(GaitTransformerConfig(
            input_dim=input_dim,
            sequence_length=window_size,
            num_classes=num_classes,
            num_heads=4,
            num_layers=2,
            embed_dim=hidden_dim // 2,
        ))
        
        # 3. 앙상블 헤드
        # LSTM 출력 + Transformer 출력 + Rule-based Feature 결합
        self.ensemble_fc = nn.Sequential(
            nn.Linear(num_classes + num_classes + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, rule_features: torch.Tensor):
        """
        x: [batch, seq_len, input_dim]
        rule_features: [batch, 1] (예: 좌우 비대칭 지수 등)
        """
        # LSTM 예측
        lstm_logits, _, _ = self.lstm_model(x)
        lstm_probs = torch.softmax(lstm_logits, dim=1)
        
        # Transformer 예측 — GaitTransformer.forward() → (logits, fall_score, attn)
        trans_logits, _fall_score, _attn = self.transformer_model(x)
        trans_probs = torch.softmax(trans_logits, dim=1)
        
        # 특징 결합
        combined = torch.cat([lstm_probs, trans_probs, rule_features], dim=1)
        
        # 최종 이상 점수 (Anomaly Score)
        anomaly_score = self.ensemble_fc(combined)
        
        return anomaly_score, {
            "lstm_probs": lstm_probs,
            "trans_probs": trans_probs,
            "anomaly_score": anomaly_score
        }

    def predict_status(self, x, rule_features: torch.Tensor, threshold: float = 0.7):
        """최종 상태 판정"""
        score, details = self.forward(x, rule_features)
        is_anomaly = score > threshold
        return is_anomaly, score, details

if __name__ == "__main__":
    # 데모 테스트
    input_dim = 6
    model = ShoeallsEnsembleModel(input_dim=input_dim)
    dummy_input = torch.randn(2, 30, input_dim)
    dummy_rules = torch.tensor([[0.1], [0.8]]) # 가상의 좌우 비대칭 지수
    
    score, info = model(dummy_input, dummy_rules)
    print(f"Final Anomaly Score: {score.flatten().tolist()}")
