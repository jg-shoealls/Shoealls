"""HuggingFace 기반 인코더 모음.

각 인코더는 기존 encoders.py와 동일한 출력 형태를 가져
MultimodalGaitNet의 CrossModalAttentionFusion에 plug-in 가능:
  - PatchTSTIMUEncoder   : (B, 6, T) -> (B, T', embed_dim)
  - VideoMAESkeletonEncoder: (B, 3, T, J) -> (B, T', embed_dim)

사전학습 가중치 다운로드 실패 시 random init으로 graceful fallback.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# ── 1. PatchTST IMU 인코더 ─────────────────────────────────────────────────────

_PATCHTST_MODEL_ID = "ibm-granite/granite-timeseries-patchtst"


class PatchTSTIMUEncoder(nn.Module):
    """PatchTST 기반 IMU 시계열 인코더.

    HuggingFace PatchTSTModel 을 IMU (6ch, T steps) 에 적용.
    패치로 분할된 시계열을 Transformer 로 인코딩한 뒤 embed_dim 으로 투영.

    Input : (B, 6, T)       — raw IMU channels-first
    Output: (B, T', embed_dim) — T' = num_patches
    """

    def __init__(
        self,
        embed_dim: int = 128,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        pretrained: bool = True,
        in_channels: int = 6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_len = patch_len
        self.stride = stride

        self.backbone = self._build_backbone(
            in_channels=in_channels,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pretrained=pretrained,
        )
        self.proj = nn.Linear(d_model, embed_dim)

    @staticmethod
    def _build_backbone(
        in_channels: int,
        patch_len: int,
        stride: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        pretrained: bool,
    ) -> nn.Module:
        try:
            from transformers import PatchTSTConfig, PatchTSTModel

            if pretrained:
                try:
                    model = PatchTSTModel.from_pretrained(
                        _PATCHTST_MODEL_ID,
                        num_input_channels=in_channels,
                        ignore_mismatched_sizes=True,
                    )
                    logger.info(f"PatchTST pretrained weights loaded from {_PATCHTST_MODEL_ID}")
                    return model
                except Exception as e:
                    logger.warning(f"Pretrained PatchTST 로드 실패, random init 사용: {e}")

            cfg = PatchTSTConfig(
                num_input_channels=in_channels,
                context_length=128,
                patch_length=patch_len,
                patch_stride=stride,
                d_model=d_model,
                num_attention_heads=num_heads,
                num_hidden_layers=num_layers,
                dropout=dropout,
                head_dropout=dropout,
            )
            logger.info("PatchTST random init (pretrained=False or offline)")
            return PatchTSTModel(cfg)

        except ImportError:
            logger.warning("transformers 없음 — PatchTST fallback Conv1D 사용")
            return _FallbackConv1DEncoder(in_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, T)  →  PatchTST expects (B, T, n_vars)
        x_t = x.permute(0, 2, 1)          # (B, T, 6)

        try:
            out = self.backbone(past_values=x_t)
            # PatchTSTModel output: last_hidden_state (B, n_vars, num_patches, d_model)
            # → mean over n_vars → (B, num_patches, d_model)
            h = out.last_hidden_state
            if h.dim() == 4:
                h = h.mean(dim=1)          # (B, num_patches, d_model)
        except Exception:
            # Fallback: Conv1D backbone returns (B, T', d_model)
            h = self.backbone(x)

        return self.proj(h)               # (B, T', embed_dim)


class _FallbackConv1DEncoder(nn.Module):
    """transformers 미설치 또는 모델 로드 실패 시 사용하는 경량 fallback."""

    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, d_model, 3, padding=1),
            nn.ReLU(), nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, T)
        return self.net(x).permute(0, 2, 1)   # (B, T', d_model)


# ── 2. VideoMAE 스켈레톤 인코더 ────────────────────────────────────────────────

_VIDEOMAE_MODEL_ID = "MCG-NJU/videomae-base"
_SKEL_IMG_SIZE = 224
_VIDEOMAE_NUM_FRAMES = 16   # VideoMAE 기본 프레임 수


class VideoMAESkeletonEncoder(nn.Module):
    """VideoMAE 기반 스켈레톤 시퀀스 인코더.

    스켈레톤 관절 좌표를 2D 스틱피겨 이미지 시퀀스로 변환 후
    VideoMAEModel 에 입력해 시공간 특징을 추출한다.

    Input : (B, 3, T, J=17)    — coords-first skeleton
    Output: (B, T', embed_dim) — T' = num_patches from VideoMAE
    """

    # COCO 17 관절 연결 (시각화용)
    _SKELETON_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),          # 얼굴
        (5, 6),                                    # 어깨
        (5, 7), (7, 9),                            # 왼팔
        (6, 8), (8, 10),                           # 오른팔
        (5, 11), (6, 12), (11, 12),               # 몸통
        (11, 13), (13, 15),                        # 왼다리
        (12, 14), (14, 16),                        # 오른다리
    ]

    def __init__(
        self,
        embed_dim: int = 128,
        img_size: int = _SKEL_IMG_SIZE,
        num_frames: int = _VIDEOMAE_NUM_FRAMES,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.num_frames = num_frames

        self.backbone, self.hidden_size, self._use_fallback = self._build_backbone(
            pretrained, img_size, num_frames
        )
        self.proj = nn.Linear(self.hidden_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_backbone(
        pretrained: bool, img_size: int, num_frames: int
    ) -> tuple[nn.Module, int, bool]:
        try:
            from transformers import VideoMAEConfig, VideoMAEModel

            if pretrained:
                try:
                    model = VideoMAEModel.from_pretrained(_VIDEOMAE_MODEL_ID)
                    logger.info(f"VideoMAE pretrained weights loaded from {_VIDEOMAE_MODEL_ID}")
                    return model, model.config.hidden_size, False
                except Exception as e:
                    logger.warning(f"Pretrained VideoMAE 로드 실패, random init 사용: {e}")

            cfg = VideoMAEConfig(
                image_size=img_size,
                num_frames=num_frames,
                num_channels=3,
                hidden_size=768,
                num_hidden_layers=4,
                num_attention_heads=8,
                intermediate_size=1536,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
            logger.info("VideoMAE random init (pretrained=False or offline)")
            return VideoMAEModel(cfg), cfg.hidden_size, False

        except ImportError:
            logger.warning("transformers 없음 — VideoMAE fallback LSTM 사용")
            fb = _FallbackLSTMEncoder(3 * 17, 768)
            return fb, 768, True

    def _skeleton_to_frames(self, skeleton: torch.Tensor) -> torch.Tensor:
        """스켈레톤 좌표 → 스틱피겨 픽셀 시퀀스.

        Args:
            skeleton: (B, 3, T, J)
        Returns:
            pixel_values: (B, num_frames, 3, img_size, img_size)
        """
        B, _, T, J = skeleton.shape
        device = skeleton.device

        # x, y 좌표만 사용 (z는 depth — 2D 이미지에선 무시)
        xy = skeleton[:, :2, :, :]   # (B, 2, T, J)

        # 시간 축을 num_frames 로 리샘플 (선형 보간)
        if T != self.num_frames:
            xy = nn.functional.interpolate(
                xy.reshape(B, 2 * J, T).float(),
                size=self.num_frames,
                mode="linear",
                align_corners=False,
            ).reshape(B, 2, self.num_frames, J)

        # 좌표를 0~1 정규화 (배치 내 전체 범위 기준)
        xy_min = xy.flatten(1).min(dim=1).values.view(B, 1, 1, 1)
        xy_max = xy.flatten(1).max(dim=1).values.view(B, 1, 1, 1)
        xy_norm = (xy.float() - xy_min) / (xy_max - xy_min + 1e-6)  # (B, 2, F, J)

        # 픽셀 좌표
        S = self.img_size
        px = (xy_norm[:, 0] * (S - 1)).long().clamp(0, S - 1)  # (B, F, J)
        py = (xy_norm[:, 1] * (S - 1)).long().clamp(0, S - 1)  # (B, F, J)

        # 빈 캔버스
        frames = torch.zeros(B, self.num_frames, 3, S, S, device=device)

        # 관절 점 그리기 (3×3 blob)
        for r in range(-1, 2):
            for c in range(-1, 2):
                yr = (py + r).clamp(0, S - 1)
                xc = (px + c).clamp(0, S - 1)
                frames[
                    torch.arange(B).unsqueeze(1).unsqueeze(2),
                    torch.arange(self.num_frames).unsqueeze(0).unsqueeze(2),
                    :,
                    yr,
                    xc,
                ] = 1.0

        # 뼈대 엣지 그리기 (선분 — 5개 중간점)
        for j1, j2 in self._SKELETON_EDGES:
            if j1 >= J or j2 >= J:
                continue
            for alpha in [0.2, 0.4, 0.6, 0.8]:
                ym = ((py[:, :, j1].float() * (1 - alpha) + py[:, :, j2].float() * alpha)
                      .long().clamp(0, S - 1))
                xm = ((px[:, :, j1].float() * (1 - alpha) + px[:, :, j2].float() * alpha)
                      .long().clamp(0, S - 1))
                frames[
                    torch.arange(B).unsqueeze(1),
                    torch.arange(self.num_frames).unsqueeze(0),
                    :,
                    ym,
                    xm,
                ] = 0.7

        return frames   # (B, F, 3, S, S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, J)
        B, C, T, J = x.shape
        if self._use_fallback:
            # LSTM fallback: (B, T, C*J)
            h = self.backbone(x.permute(0, 2, 1, 3).reshape(B, T, C * J))
        else:
            try:
                pixel_values = self._skeleton_to_frames(x)   # (B, F, 3, S, S)
                out = self.backbone(pixel_values=pixel_values)
                h = out.last_hidden_state                     # (B, num_patches, hidden_size)
            except Exception as e:
                logger.warning(f"VideoMAE forward 실패 ({e}), mean-pool fallback 사용")
                # 최후 fallback: 관절 좌표 평균풀링
                h = x.mean(dim=-1).permute(0, 2, 1)           # (B, T, C)
                h = nn.functional.adaptive_avg_pool1d(
                    h.permute(0, 2, 1), 8
                ).permute(0, 2, 1)                            # (B, 8, C)
                # hidden_size 맞춤
                h = h.expand(-1, -1, self.hidden_size // C * C)

        h = self.dropout(h)
        return self.proj(h)                               # (B, T', embed_dim)


class _FallbackLSTMEncoder(nn.Module):
    """transformers 미설치 또는 VideoMAE 로드 실패 시 경량 fallback."""

    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_size // 2, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_features)
        out, _ = self.lstm(x)
        return out   # (B, T, hidden_size)
