"""Modality-specific encoders for multimodal gait analysis."""

import torch
import torch.nn as nn


class IMUEncoder(nn.Module):
    """1D-CNN + LSTM encoder for IMU time-series data.

    Input shape: (batch, channels=6, time)
    Output shape: (batch, time', embed_dim)
    """

    def __init__(
        self,
        in_channels: int = 6,
        conv_channels: list = None,
        kernel_size: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        conv_channels = conv_channels or [32, 64, 128]

        # Build 1D convolution layers
        layers = []
        ch_in = in_channels
        for ch_out in conv_channels:
            layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            ch_in = ch_out
        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )

        self.proj = nn.Linear(lstm_hidden * 2, lstm_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, T)
        features = self.cnn(x)          # (B, C, T')
        features = features.permute(0, 2, 1)  # (B, T', C)
        lstm_out, _ = self.lstm(features)      # (B, T', 2*hidden)
        return self.proj(lstm_out)             # (B, T', hidden)


class PressureEncoder(nn.Module):
    """2D-CNN encoder for plantar pressure maps.

    Input shape: (batch, time, 1, H, W)
    Output shape: (batch, time, embed_dim)
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: list = None,
        kernel_size: int = 3,
        embed_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        conv_channels = conv_channels or [16, 32, 64]

        layers = []
        ch_in = in_channels
        for ch_out in conv_channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            ])
            ch_in = ch_out
        self.cnn = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(conv_channels[-1], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W)
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])    # (B*T, 1, H, W)
        features = self.cnn(x)                  # (B*T, C, H', W')
        features = self.adaptive_pool(features) # (B*T, C, 1, 1)
        features = features.flatten(1)          # (B*T, C)
        features = self.proj(features)          # (B*T, embed_dim)
        return features.reshape(B, T, -1)       # (B, T, embed_dim)


class SkeletonEncoder(nn.Module):
    """Spatial-Temporal Graph Convolution encoder for skeleton data.

    Input shape: (batch, coords=3, time, joints)
    Output shape: (batch, time', embed_dim)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_joints: int = 17,
        gcn_channels: list = None,
        temporal_kernel: int = 3,
        embed_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        gcn_channels = gcn_channels or [64, 128]

        # Build adjacency matrix for human skeleton
        self.num_joints = num_joints
        self.register_buffer("adj", self._build_adjacency(num_joints))

        # Spatial-temporal convolution blocks
        blocks = []
        ch_in = in_channels
        for ch_out in gcn_channels:
            blocks.append(
                STGCNBlock(ch_in, ch_out, num_joints, temporal_kernel, dropout)
            )
            ch_in = ch_out
        self.st_blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(gcn_channels[-1], embed_dim)

    @staticmethod
    def _build_adjacency(num_joints: int) -> torch.Tensor:
        """Build skeleton adjacency matrix."""
        edges = [
            (0, 1), (1, 2),           # hip -> spine -> head
            (1, 3), (3, 4), (4, 5),   # spine -> L.shoulder -> L.elbow -> L.wrist
            (1, 6), (6, 7), (7, 8),   # spine -> R.shoulder -> R.elbow -> R.wrist
            (0, 9), (9, 10), (10, 11),  # hip -> L.hip -> L.knee -> L.ankle
            (0, 12), (12, 13), (13, 14),  # hip -> R.hip -> R.knee -> R.ankle
        ]
        if num_joints > 15:
            edges.extend([(11, 15), (14, 16)])  # ankles -> feet

        adj = torch.eye(num_joints)
        for i, j in edges:
            if i < num_joints and j < num_joints:
                adj[i, j] = 1
                adj[j, i] = 1

        # Normalize
        degree = adj.sum(dim=1, keepdim=True)
        adj = adj / degree
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, J)
        B, C, T, J = x.shape

        for block in self.st_blocks:
            x = block(x, self.adj)

        # Global joint pooling: (B, C', T, J) -> (B, C', T)
        x = x.mean(dim=-1)

        # Project to embedding dim: (B, C', T) -> (B, T, embed_dim)
        x = x.permute(0, 2, 1)
        return self.proj(x)


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int,
        temporal_kernel: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Spatial graph convolution
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.spatial_bn = nn.BatchNorm2d(out_channels)

        # Temporal convolution
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(temporal_kernel, 1),
            padding=(temporal_kernel // 2, 0),
        )
        self.temporal_bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, J)
        residual = self.residual(x)

        # Spatial graph convolution: multiply by adjacency
        # (B, C, T, J) @ (J, J) -> (B, C, T, J)
        x = torch.einsum("bctj,jk->bctk", x, adj)
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.relu(x)

        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)

        x = self.relu(x + residual)
        return self.dropout(x)
