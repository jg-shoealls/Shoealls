"""Cross-modal attention fusion for multimodal gait features."""

import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    """Fuse features from multiple modalities using cross-modal attention.

    Each modality attends to all other modalities, then features are combined.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 2,
        num_modalities: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_modalities = num_modalities

        # Modality-specific tokens
        self.modality_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            for _ in range(num_modalities)
        ])

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Self-attention for fused representation
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: List of tensors, each (B, T_i, D).

        Returns:
            Fused representation of shape (B, D).
        """
        # Add modality embeddings
        enriched = []
        for i, (feat, mod_emb) in enumerate(
            zip(modality_features, self.modality_embeddings)
        ):
            enriched.append(feat + mod_emb.expand(feat.size(0), feat.size(1), -1))

        # Cross-modal attention: each modality attends to concatenation of others
        for layer in self.cross_attention_layers:
            updated = []
            for i in range(self.num_modalities):
                # Concatenate all other modalities as context
                context_parts = [enriched[j] for j in range(self.num_modalities) if j != i]
                context = torch.cat(context_parts, dim=1)
                updated.append(layer(enriched[i], context))
            enriched = updated

        # Concatenate all modality representations
        combined = torch.cat(enriched, dim=1)  # (B, sum(T_i), D)

        # Self-attention over combined
        # Optimized: need_weights=False disables attention weight calculation/allocation
        # and enables optimized paths (e.g., FlashAttention)
        attn_out, _ = self.self_attention(combined, combined, combined, need_weights=False)
        combined = self.norm(combined + attn_out)

        # Global average pooling
        return combined.mean(dim=1)  # (B, D)


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block with feed-forward network."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross-attention: query attends to context
        # Optimized: need_weights=False disables attention weight calculation/allocation
        attn_out, _ = self.cross_attn(query, context, context, need_weights=False)
        query = self.norm1(query + attn_out)

        # Feed-forward
        ff_out = self.ffn(query)
        return self.norm2(query + ff_out)
