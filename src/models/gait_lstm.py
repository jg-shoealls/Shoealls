"""
LSTM 기반 보행 분류 모델 + 룰 기반 낙상 감지 통합.

CES 데이터 스키마 기반:
  - 입력 특성: speed, tilt, fe_event_value, event_type_encoded, foot_type_enc, confidence_norm
  - 출력 클래스: 0=normal, 1=waiting, 2=drag
  - 낙상(fall): fe_event_type == 'fall' → 룰 기반 즉각 알림 + AI 위험 점수 병행
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

# ─── Dataset ─────────────────────────────────────────────────────────────────


class GaitWindowDataset(Dataset):
    """
    슬라이딩 윈도우 기반 시계열 Dataset.

    각 샘플: {"x": [window_size, n_features], "label": int, "fall_flag": bool}
    fall_flag: 윈도우 내 fall 이벤트 존재 여부 (룰 기반 즉각 알림용)
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        fall_flags: np.ndarray | None = None,
        window_size: int = 30,
        stride: int = 10,
    ) -> None:
        if features.ndim != 2:
            raise ValueError("features must be a 2D array: [rows, feature_dim]")
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same row count")
        if fall_flags is not None and len(fall_flags) != len(features):
            raise ValueError("fall_flags and features must have the same row count")
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if len(features) < window_size:
            raise ValueError("not enough rows to create one time-series window")

        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.fall_flags = (
            fall_flags.astype(np.int64)
            if fall_flags is not None
            else np.zeros(len(features), dtype=np.int64)
        )
        self.window_size = window_size
        self.indices = list(range(0, len(features) - window_size + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.window_size
        return {
            "x": torch.from_numpy(self.features[start:end]),
            "label": torch.tensor(int(self.labels[end - 1]), dtype=torch.long),
            "fall_flag": torch.tensor(
                int(self.fall_flags[start:end].max()), dtype=torch.bool
            ),
        }


# 하위 호환 별칭
GaitDataset = GaitWindowDataset


# ─── 어텐션 ──────────────────────────────────────────────────────────────────


class TemporalAttention(nn.Module):
    """LSTM 시퀀스 출력에 대한 소프트 어텐션 — 중요한 시간 구간 강조"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, lstm_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # lstm_out: [batch, seq, hidden]
        scores = self.attn(lstm_out).squeeze(-1)            # [batch, seq]
        weights = F.softmax(scores, dim=-1)                 # [batch, seq]
        context = (lstm_out * weights.unsqueeze(-1)).sum(1) # [batch, hidden]
        return context, weights


# ─── 모델 ────────────────────────────────────────────────────────────────────


class GaitLSTM(nn.Module):
    """
    보행 상태 분류 BiLSTM + Temporal Attention.

    입력:  [batch, window_size, input_dim]
    출력:  logits [batch, num_classes],  fall_score [batch, 1],  attn_weights [batch, seq]

    기존 API(`predict_with_fall_rule`)를 완전히 유지하면서 BiLSTM + Attention을 추가.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.input_norm = nn.BatchNorm1d(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.attention = TemporalAttention(lstm_out_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

        # 낙상 위험 점수 헤드 (AI 기반, 룰 기반과 OR 조합)
        self.fall_risk_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 입력 정규화: BatchNorm1d → [batch, feature, seq] 변환 후 역변환
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)               # [batch, seq, hidden*2]
        context, attn_w = self.attention(lstm_out)

        logits = self.classifier(context)         # [batch, n_classes]
        fall_score = self.fall_risk_head(context) # [batch, 1]
        return logits, fall_score, attn_w

    def predict_with_fall_rule(
        self,
        x: torch.Tensor,
        fall_flag: bool | torch.Tensor = False,
        fall_threshold: float = 0.65,
    ) -> dict[str, float | int | bool]:
        """
        AI 분류 결과 + 룰 기반 낙상 감지를 결합한 통합 추론.

        fall_flag=True (fe_event_type == 'fall') → 즉각 알림, AI 점수 무관
        fall_score >= threshold → AI 기반 고위험 알림
        """
        self.eval()
        with torch.no_grad():
            logits, fall_score_t, attn_w = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            confidence, prediction = torch.max(probs, dim=-1)

        rule_triggered = (
            bool(fall_flag.item())
            if isinstance(fall_flag, torch.Tensor)
            else bool(fall_flag)
        )
        ai_fall_score = float(fall_score_t[0, 0].item())
        ai_triggered = ai_fall_score >= fall_threshold

        return {
            "prediction_idx": int(prediction[0].item()),
            "confidence": float(confidence[0].item()),
            "probabilities": probs[0].cpu().tolist(),
            "fall_score_ai": ai_fall_score,
            "fall_alert": rule_triggered or ai_triggered,
            "is_rule_triggered": rule_triggered,      # fe_event_type == 'fall'
            "is_ai_triggered": ai_triggered,           # 모델 기반 위험 감지
            "attention_weights": attn_w[0].cpu().tolist() if attn_w is not None else [],
        }


# ─── 손실 함수 ───────────────────────────────────────────────────────────────


class CombinedGaitLoss(nn.Module):
    """
    다중 태스크 손실:
      total = CrossEntropy(분류) + lambda_fall × BCE(낙상 위험)

    낙상은 희소하므로 lambda_fall > 1로 상향 가중치 적용.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        lambda_fall: float = 2.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.bce = nn.BCELoss()
        self.lambda_fall = lambda_fall

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        fall_score: torch.Tensor,
        fall_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss = self.ce(logits, labels)
        bce_loss = self.bce(fall_score.squeeze(-1), fall_targets.float())
        total = ce_loss + self.lambda_fall * bce_loss
        return total, {
            "ce": float(ce_loss.item()),
            "bce": float(bce_loss.item()),
            "total": float(total.item()),
        }


# ─── 학습 루프 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    accuracy: float
    fall_bce: float = 0.0


def train_one_epoch(
    model: GaitLSTM,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> TrainStepResult:
    """1 에폭 학습 루프"""
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        label = batch["label"].to(device)
        fall_flag = batch["fall_flag"].to(device).long()

        optimizer.zero_grad(set_to_none=True)
        logits, fall_score, _ = model(x)

        if isinstance(criterion, CombinedGaitLoss):
            loss, breakdown = criterion(logits, label, fall_score, fall_flag)
            total_bce += breakdown["bce"]
        else:
            loss = criterion(logits, label)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        correct += int((logits.argmax(dim=-1) == label).sum().item())
        total += int(label.numel())

    n = max(len(dataloader), 1)
    return TrainStepResult(
        loss=total_loss / n,
        accuracy=correct / max(total, 1),
        fall_bce=total_bce / n,
    )


@torch.no_grad()
def evaluate(
    model: GaitLSTM,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """검증 루프: accuracy, fall recall 반환"""
    model.eval()
    correct = 0
    total = 0
    fall_correct = 0
    fall_total = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        label = batch["label"].to(device)
        fall_flag = batch["fall_flag"].to(device)

        logits, fall_score, _ = model(x)
        preds = logits.argmax(dim=-1)
        correct += int((preds == label).sum().item())
        total += int(label.numel())

        # 낙상 감지 recall (룰 기반 fall_flag 기준)
        predicted_fall = (fall_score.squeeze(-1) >= 0.65)
        fall_total += int(fall_flag.sum().item())
        fall_correct += int((predicted_fall & fall_flag.bool()).sum().item())

    return {
        "accuracy": correct / max(total, 1),
        "fall_recall": fall_correct / max(fall_total, 1),
    }


# ─── 빠른 형상 검증 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = GaitLSTM(input_dim=6, hidden_dim=128, num_classes=3)
    dummy = torch.randn(4, 30, 6)
    logits, fall_score, attn = model(dummy)

    print(f"logits:     {logits.shape}")      # [4, 3]
    print(f"fall_score: {fall_score.shape}")  # [4, 1]
    print(f"attn:       {attn.shape}")        # [4, 30]

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"파라미터 수: {n_params:,}")

    # predict_with_fall_rule 테스트
    result = model.predict_with_fall_rule(dummy[:1], fall_flag=False)
    print("\n추론 결과:", result)
