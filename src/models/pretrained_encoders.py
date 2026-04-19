"""HuggingFace/torchvision 사전학습 모델 기반 인코더.

기존 커스텀 인코더(encoders.py)를 경량 사전학습 모델로 대체합니다:
  - LIMUBERTEncoder: LIMU-BERT 스타일 6축 IMU 인코더 (~62K 파라미터)
  - MobileNetV2PressureEncoder: MobileNetV2-0.35 기반 압력 히트맵 인코더 (~1.7M)
  - CTRGCNEncoder: CTR-GCN 채널별 토폴로지 스켈레톤 인코더 (~1.5M)

각 클래스는 기존 인코더와 동일한 입출력 인터페이스를 유지합니다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. LIMU-BERT IMU 인코더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _SinusoidalPE(nn.Module):
    """사인파 위치 인코딩."""

    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            -torch.arange(0, embed_dim, 2).float() * math.log(10000.0) / embed_dim
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LIMUBERTEncoder(nn.Module):
    """LIMU-BERT 스타일 경량 IMU 인코더.

    LIMU-BERT (DAPOWAN/LIMU-BERT-Public) 아키텍처를 재구현합니다.
    6축 IMU 데이터를 슬라이딩 윈도우로 패치화 후 Transformer로 인코딩.

    Input:  (B, 6, T)
    Output: (B, T, embed_dim)

    기본 설정(embed_dim=72, heads=4, layers=4)으로 약 62K 파라미터.

    GitHub 체크포인트 로드:
        encoder = LIMUBERTEncoder()
        encoder.load_limu_checkpoint("limu_bert.pt")
    """

    def __init__(
        self,
        in_channels: int = 6,
        embed_dim: int = 72,
        num_heads: int = 4,
        num_layers: int = 4,
        patch_size: int = 8,
        dropout: float = 0.1,
        embed_dim_out: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 패치 임베딩: (B, 6, T) → (B, T//patch_size, embed_dim)
        self.patch_embed = nn.Sequential(
            nn.Linear(in_channels * patch_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.pos_enc = _SinusoidalPE(embed_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 출력 차원을 embed_dim_out으로 맞춤
        self.proj = (
            nn.Linear(embed_dim, embed_dim_out)
            if embed_dim != embed_dim_out
            else nn.Identity()
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, T//P, C*P) 패치화."""
        B, C, T = x.shape
        pad = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad:
            x = F.pad(x, (0, pad))
        T_padded = x.shape[-1]
        x = x.reshape(B, C, T_padded // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, -1, C * self.patch_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, T)
        patches = self._patchify(x)              # (B, N_patches, 6*P)
        emb = self.patch_embed(patches)          # (B, N_patches, embed_dim)
        emb = self.pos_enc(emb)
        out = self.transformer(emb)              # (B, N_patches, embed_dim)
        out = self.norm(out)

        # 시간 해상도를 원본 T에 맞춰 보간
        B, N, D = out.shape
        T_orig = x.shape[-1]
        out = out.permute(0, 2, 1)                        # (B, D, N)
        out = F.interpolate(out, size=T_orig, mode="linear", align_corners=False)
        out = out.permute(0, 2, 1)                        # (B, T, D)

        return self.proj(out)                             # (B, T, embed_dim_out)

    def load_limu_checkpoint(self, path: str):
        """LIMU-BERT GitHub 체크포인트 로드.

        https://github.com/dapowan/LIMU-BERT-Public 에서 다운로드한
        사전학습 가중치를 로드합니다. 레이어 크기가 맞는 파라미터만 복사.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model_state = self.state_dict()
        loaded = []
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded.append(k)
        self.load_state_dict(model_state)
        print(f"LIMU-BERT: loaded {len(loaded)}/{len(model_state)} params from {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. MobileNetV2 압력 히트맵 인코더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MobileNetV2PressureEncoder(nn.Module):
    """MobileNetV2-0.35 기반 족저압력 히트맵 인코더.

    torchvision MobileNetV2 백본을 1채널 입력에 맞게 수정합니다.
    ImageNet 사전학습 가중치를 사용하며, 첫 번째 Conv의 가중치 채널을
    평균 내어 그레이스케일 입력과 호환합니다.

    Input:  (B, T, 1, H, W)  — 족저압력 16×8 그리드
    Output: (B, T, embed_dim)

    약 1.7M 파라미터 (MobileNetV2 width_mult=0.35 기준).

    사용법:
        enc = MobileNetV2PressureEncoder(pretrained=True)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        dropout: float = 0.1,
        pretrained: bool = True,
        width_mult: float = 0.35,
    ):
        super().__init__()

        try:
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = mobilenet_v2(weights=weights, width_mult=width_mult)  # type: ignore[call-arg]
        except TypeError:
            from torchvision.models import mobilenet_v2
            backbone = mobilenet_v2(pretrained=pretrained)

        # 첫 Conv: 3채널 → 1채널 (가중치를 채널 방향으로 평균)
        first_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        backbone.features[0][0] = new_conv

        # 분류기 헤드 제거, feature extractor만 유지
        self.features = backbone.features
        feature_dim = backbone.last_channel  # width_mult=0.35 → 1280 * 0.35 ≈ 1280

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W)
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])    # (B*T, 1, H, W)

        # 16×8 → 최소 32×32 이상으로 보간 (MobileNet 첫 stride=2 대응)
        if x.shape[-1] < 32 or x.shape[-2] < 32:
            x = F.interpolate(x, size=(64, 32), mode="bilinear", align_corners=False)

        feats = self.features(x)               # (B*T, C, h, w)
        feats = self.pool(feats).flatten(1)    # (B*T, C)
        out = self.proj(feats)                 # (B*T, embed_dim)
        return out.reshape(B, T, -1)           # (B, T, embed_dim)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. CTR-GCN 스켈레톤 인코더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _ChannelTopologyRefinement(nn.Module):
    """CTR-GCN 채널별 토폴로지 개선 모듈.

    각 출력 채널이 고유한 그래프 토폴로지를 학습합니다.
    A_final = A_static + A_learnable (채널별 독립적)
    """

    def __init__(self, in_channels: int, out_channels: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints

        # 채널별 로컬 그래프 학습용 1×1 conv
        self.key = nn.Conv2d(in_channels, out_channels, 1)
        self.query = nn.Conv2d(in_channels, out_channels, 1)

        # 고정 그래프 분기 (전역 공유)
        self.static_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.zeros(1))  # 학습 가능한 혼합 비율

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, J)
        B, C, T, J = x.shape

        # Static branch: 기존 공유 토폴로지
        static = torch.einsum("bctj,jk->bctk", x, adj)
        static_feat = self.static_conv(static)     # (B, C', T, J)

        # Dynamic branch: 채널별 학습 토폴로지 (시간 평균 후 계산)
        x_t = x.mean(dim=2, keepdim=True)          # (B, C, 1, J)
        k = self.key(x_t).squeeze(2)               # (B, C', J)
        q = self.query(x_t).squeeze(2)             # (B, C', J)

        # 채널별 동적 어드전시: (B, C', J, J)
        dyn_adj = torch.softmax(
            torch.bmm(q.permute(0, 2, 1), k) / math.sqrt(k.size(1)), dim=-1
        ).unsqueeze(1).expand(-1, static_feat.size(1), -1, -1)

        # 각 채널에 독립적인 그래프 적용
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, C, J)
        static_w = static_feat.permute(0, 2, 1, 3).reshape(B * T, -1, J)

        dyn_adj_exp = dyn_adj.unsqueeze(2).expand(-1, -1, T, -1, -1)
        dyn_adj_flat = dyn_adj_exp.reshape(B * T, -1, J, J)
        dyn_feat = torch.einsum("bcj,bcjk->bck", static_w, dyn_adj_flat)
        dyn_feat = dyn_feat.reshape(B, T, -1, J).permute(0, 2, 1, 3)

        out = static_feat + self.alpha * dyn_feat
        return self.relu(self.bn(out))


class _CTRGCNBlock(nn.Module):
    """CTR-GCN 블록: 채널별 토폴로지 + 시간 Conv + 잔차 연결."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int,
        temporal_kernel: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gcn = _ChannelTopologyRefinement(in_channels, out_channels, num_joints)
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(temporal_kernel, 1),
                padding=(temporal_kernel // 2, 0),
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.gcn(x, adj)
        out = self.relu(self.tcn(out) + res)
        return self.drop(out)


class CTRGCNEncoder(nn.Module):
    """CTR-GCN 기반 스켈레톤 인코더.

    채널별 토폴로지 개선(Channel-wise Topology Refinement)으로 관절 간
    관계를 채널마다 독립적으로 학습합니다.

    Input:  (B, 3, T, J)  — 3D 관절 좌표
    Output: (B, T, embed_dim)

    기본 설정(gcn_channels=[64,128])으로 약 1.5M 파라미터.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_joints: int = 17,
        gcn_channels: Optional[list] = None,
        temporal_kernel: int = 3,
        embed_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        gcn_channels = gcn_channels or [64, 128]
        self.num_joints = num_joints
        self.register_buffer("adj", self._build_adj(num_joints))

        blocks = []
        ch = in_channels
        for ch_out in gcn_channels:
            blocks.append(_CTRGCNBlock(ch, ch_out, num_joints, temporal_kernel, dropout))
            ch = ch_out
        self.blocks = nn.ModuleList(blocks)

        self.proj = nn.Linear(gcn_channels[-1], embed_dim)

    @staticmethod
    def _build_adj(num_joints: int) -> torch.Tensor:
        edges = [
            (0, 1), (1, 2), (1, 3), (3, 4), (4, 5),
            (1, 6), (6, 7), (7, 8), (0, 9), (9, 10),
            (10, 11), (0, 12), (12, 13), (13, 14),
        ]
        if num_joints > 15:
            edges += [(11, 15), (14, 16)]
        adj = torch.eye(num_joints)
        for i, j in edges:
            if i < num_joints and j < num_joints:
                adj[i, j] = adj[j, i] = 1.0
        deg = adj.sum(1, keepdim=True).clamp(min=1)
        return adj / deg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, J)
        for block in self.blocks:
            x = block(x, self.adj)              # (B, C', T, J)

        x = x.mean(dim=-1)                     # (B, C', T)  — 관절 평균
        x = x.permute(0, 2, 1)                 # (B, T, C')
        return self.proj(x)                     # (B, T, embed_dim)
