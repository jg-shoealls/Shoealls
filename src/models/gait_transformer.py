"""
GaitTransformer — PatchTST 기반 보행 분류 Transformer.

GaitLSTM과 동일한 인터페이스(predict_with_fall_rule, forward)를 구현하여
학습된 모델을 교체 없이 스왑(drop-in replacement) 가능.

핵심 차이:
  GaitLSTM    : BiLSTM + Temporal Attention  입력 [batch, seq, features]
  GaitTransformer: PatchTST (패치 기반 Transformer) 입력 [batch, seq, features] → 내부 전치

인터페이스:
  forward(x) → (logits [batch, n_classes], fall_score [batch, 1], attn_weights [batch, n_patches+1, n_patches+1])
  predict_with_fall_rule(x, fall_flag) → dict (GaitLSTM 반환 형태 동일)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.har_transformer import PatchTSTClassifier

logger = logging.getLogger(__name__)


# ─── 설정 ────────────────────────────────────────────────────────────────────


@dataclass
class GaitTransformerConfig:
    input_dim: int = 6           # 특성 수 (GaitLSTM 호환 파라미터명)
    sequence_length: int = 30    # 윈도우 크기
    patch_size: int = 5          # 패치 크기 (sequence_length의 약수 권장)
    stride: int = 2              # 패치 스트라이드
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 128
    dropout: float = 0.1
    num_classes: int = 3


# ─── 낙상 위험 헤드 ──────────────────────────────────────────────────────────


class FallRiskHead(nn.Module):
    """PatchTST 임베딩에서 낙상 위험 점수(0~1) 산출"""

    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── GaitTransformer ─────────────────────────────────────────────────────────


class GaitTransformer(nn.Module):
    """
    PatchTST 기반 보행 분류 Transformer.

    GaitLSTM과 완전 동일한 공개 인터페이스:
      - forward(x) → (logits, fall_score, attn_weights)
      - predict_with_fall_rule(x, fall_flag, fall_threshold) → dict

    모델 교체 예시:
        # 기존 코드
        model = GaitLSTM(input_dim=6)
        # Transformer로 교체 (인터페이스 동일)
        model = GaitTransformer(GaitTransformerConfig(input_dim=6))
    """

    def __init__(self, cfg: GaitTransformerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GaitTransformerConfig()

        # PatchTST 백본: [batch, channels, time] 입력
        self.backbone = PatchTSTClassifier(
            in_channels=self.cfg.input_dim,
            num_classes=self.cfg.num_classes,
            sequence_length=self.cfg.sequence_length,
            patch_size=self.cfg.patch_size,
            stride=self.cfg.stride,
            embed_dim=self.cfg.embed_dim,
            num_heads=self.cfg.num_heads,
            num_layers=self.cfg.num_layers,
            ff_dim=self.cfg.ff_dim,
            dropout=self.cfg.dropout,
        )

        # 낙상 위험 헤드 (백본 피처 재사용)
        self.fall_head = FallRiskHead(self.cfg.embed_dim, self.cfg.dropout)

        # 어텐션 저장용 훅
        self._attn_cache: list[torch.Tensor] = []
        self._register_attn_hook()

        logger.info(
            "GaitTransformer 초기화: input=%d seq=%d patches=%d params=%d",
            self.cfg.input_dim, self.cfg.sequence_length,
            self.backbone.num_patches, self._count_params(),
        )

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _register_attn_hook(self) -> None:
        """마지막 Transformer 레이어의 self-attention 가중치 캐싱"""
        def hook(module: nn.Module, input: tuple, output: tuple) -> None:
            # TransformerEncoderLayer의 self_attn은 (output, attn_weights) 반환
            # attn_output_weights가 있을 때만 저장
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self._attn_cache.clear()
                self._attn_cache.append(output[1].detach())

        last_layer = self.backbone.encoder.layers[-1]
        last_layer.self_attn.register_forward_hook(hook)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: [batch, seq, features]  ← GaitLSTM과 동일 입력 형태
        Returns:
            logits:       [batch, n_classes]
            fall_score:   [batch, 1]
            attn_weights: [batch, n_heads, seq, seq] 또는 None
        """
        # PatchTST는 [batch, channels, time] 기대 → 전치
        x_t = x.transpose(1, 2)  # [batch, features, seq]

        # 시퀀스 길이 불일치 처리 (추론 시 다른 길이 입력 허용)
        if x_t.size(2) != self.cfg.sequence_length:
            x_t = F.interpolate(
                x_t, size=self.cfg.sequence_length, mode="linear", align_corners=False
            )

        self._attn_cache.clear()
        features = self.backbone.forward_features(x_t)  # [batch, embed_dim]
        logits = self.backbone.classifier(features)      # [batch, n_classes]
        fall_score = self.fall_head(features)             # [batch, 1]

        attn = self._attn_cache[0] if self._attn_cache else None
        return logits, fall_score, attn

    def predict_with_fall_rule(
        self,
        x: torch.Tensor,
        fall_flag: bool | torch.Tensor = False,
        fall_threshold: float = 0.65,
    ) -> dict:
        """
        GaitLSTM.predict_with_fall_rule과 동일 인터페이스.
        CE 앙상블 모델(ces_ensemble.py)이 이 메서드를 호출하므로 교체 가능.
        """
        self.eval()
        with torch.no_grad():
            logits, fall_score_t, attn = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence, prediction = torch.max(probs, dim=-1)

        rule_triggered = (
            bool(fall_flag.item()) if isinstance(fall_flag, torch.Tensor) else bool(fall_flag)
        )
        ai_fall_score = float(fall_score_t[0, 0].item())
        ai_triggered = ai_fall_score >= fall_threshold

        # 어텐션 가중치 평균 (헤드 평균 → CLS 토큰 기준)
        attn_list: list[float] | None = None
        if attn is not None:
            # [batch, heads, seq, seq] → CLS(0) 행 평균
            attn_cls = attn[0].mean(0)[0].cpu().tolist()  # [seq]
            attn_list = attn_cls

        return {
            "prediction_idx": int(prediction[0].item()),
            "confidence": float(confidence[0].item()),
            "probabilities": probs[0].cpu().tolist(),
            "fall_score_ai": ai_fall_score,
            "fall_alert": rule_triggered or ai_triggered,
            "is_rule_triggered": rule_triggered,
            "is_ai_triggered": ai_triggered,
            "attention_weights": attn_list or [],
        }


# ─── 스왑 팩토리 ──────────────────────────────────────────────────────────────


def build_gait_model(
    backend: str = "lstm",
    input_dim: int = 6,
    hidden_dim: int = 128,
    num_classes: int = 3,
    sequence_length: int = 30,
    dropout: float = 0.3,
) -> nn.Module:
    """
    백엔드 문자열로 GaitLSTM / GaitTransformer를 생성하는 팩토리.
    동일 파라미터 명세로 양쪽 모두 생성 가능.

    사용 예시:
        model = build_gait_model(backend="transformer", input_dim=6)
        # 이후 코드는 model.predict_with_fall_rule(...) 그대로 동작
    """
    if backend == "transformer":
        cfg = GaitTransformerConfig(
            input_dim=input_dim,
            sequence_length=sequence_length,
            embed_dim=min(hidden_dim, 96),
            num_classes=num_classes,
            dropout=dropout,
        )
        return GaitTransformer(cfg)

    # 기본: LSTM
    from src.models.gait_lstm import GaitLSTM
    return GaitLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )


# ─── 체크포인트 로더 (GaitLSTM ↔ GaitTransformer 교차 로드 방지) ───────────────


def load_gait_checkpoint(path: str, backend: str = "auto") -> tuple[nn.Module, dict]:
    """
    저장된 체크포인트에서 모델 유형을 자동 감지하여 로드.

    Returns:
        model: 로드된 모델
        meta:  체크포인트 메타데이터
    """
    ckpt = torch.load(path, map_location="cpu")
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}

    model_type = meta.get("model_type", "lstm")
    if backend != "auto":
        model_type = backend

    n_features = meta.get("n_features", 6)
    hidden_dim = meta.get("hidden_dim", 128)
    window_size = meta.get("window_size", 30)
    n_classes = meta.get("n_classes", 3)

    model = build_gait_model(
        backend="transformer" if model_type == "transformer" else "lstm",
        input_dim=n_features,
        hidden_dim=hidden_dim,
        num_classes=n_classes,
        sequence_length=window_size,
    )

    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("누락된 키: %s", missing[:5])
    if unexpected:
        logger.warning("예상치 못한 키: %s", unexpected[:5])

    logger.info("체크포인트 로드: %s → %s", path, model.__class__.__name__)
    return model, meta


# ─── 빠른 형상 검증 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    for backend in ["lstm", "transformer"]:
        model = build_gait_model(backend=backend, input_dim=6, sequence_length=30)
        dummy = torch.randn(4, 30, 6)
        logits, fall_score, attn = model(dummy)
        print(f"\n[{backend.upper()}]")
        print(f"  logits:     {logits.shape}")
        print(f"  fall_score: {fall_score.shape}")
        print(f"  attn:       {attn.shape if attn is not None else None}")
        print(f"  params:     {sum(p.numel() for p in model.parameters()):,}")

        result = model.predict_with_fall_rule(dummy[:1], fall_flag=False)
        print(f"  predict:    {result['prediction_idx']} conf={result['confidence']:.3f}")
