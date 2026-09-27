"""Lightweight Transformer classifiers for wearable HAR windows."""

from __future__ import annotations

import torch
from torch import nn


class PatchTSTClassifier(nn.Module):
    """Compact PatchTST-style classifier for ``(batch, channels, time)`` input."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        sequence_length: int = 128,
        patch_size: int = 16,
        stride: int = 8,
        embed_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_size > sequence_length:
            raise ValueError("patch_size must be <= sequence_length")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = 1 + (sequence_length - patch_size) // stride

        self.patch_proj = nn.Linear(in_channels * patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled sequence representation before the classifier head."""
        if x.ndim != 3:
            raise ValueError("expected input shape (batch, channels, time)")
        if x.size(1) != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x.size(1)}")
        if x.size(2) != self.sequence_length:
            raise ValueError(
                f"expected sequence length {self.sequence_length}, got {x.size(2)}"
            )

        patches = x.unfold(dimension=2, size=self.patch_size, step=self.stride)
        patches = patches.permute(0, 2, 1, 3).flatten(2)
        tokens = self.patch_proj(patches)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def freeze_encoder(model: PatchTSTClassifier) -> None:
    """Freeze all representation layers and leave only the classifier trainable."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")
