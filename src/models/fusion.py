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

    def forward(
        self,
        modality_features: list[torch.Tensor],
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Args:
            modality_features: List of tensors, each (B, T_i, D).
            return_attn_weights: If True, also return cross-attention weight maps.

        Returns:
            Fused representation of shape (B, D).
            If return_attn_weights: also returns dict with attention weight info.
        """
        # Add modality embeddings
        enriched = []
        for i, (feat, mod_emb) in enumerate(
            zip(modality_features, self.modality_embeddings)
        ):
            enriched.append(feat + mod_emb.expand(feat.size(0), feat.size(1), -1))

        # Cross-modal attention: each modality attends to concatenation of others
        all_attn_weights = []  # layer -> modality -> (B, heads, T_q, T_ctx)
        for layer in self.cross_attention_layers:
            updated = []
            layer_weights = []
            for i in range(self.num_modalities):
                # Concatenate all other modalities as context
                context_parts = [enriched[j] for j in range(self.num_modalities) if j != i]
                context = torch.cat(context_parts, dim=1)
                if return_attn_weights:
                    out, weights = layer(enriched[i], context, return_attn_weights=True)
                    updated.append(out)
                    layer_weights.append(weights)
                else:
                    updated.append(layer(enriched[i], context))
            enriched = updated
            if return_attn_weights:
                all_attn_weights.append(layer_weights)

        # Concatenate all modality representations
        combined = torch.cat(enriched, dim=1)  # (B, sum(T_i), D)

        # Self-attention over combined
        attn_out, self_attn_w = self.self_attention(
            combined, combined, combined,
            need_weights=return_attn_weights,
            average_attn_weights=False,
        )
        combined = self.norm(combined + attn_out)

        # Global average pooling
        fused = combined.mean(dim=1)  # (B, D)

        if return_attn_weights:
            attn_info = {
                "cross_attn_weights": all_attn_weights,
                "self_attn_weights": self_attn_w,
            }
            return fused, attn_info
        return fused


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

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Cross-attention: query attends to context
        attn_out, attn_weights = self.cross_attn(
            query, context, context,
            need_weights=return_attn_weights,
            average_attn_weights=False,
        )
        query = self.norm1(query + attn_out)

        # Feed-forward
        ff_out = self.ffn(query)
        out = self.norm2(query + ff_out)

        if return_attn_weights:
            return out, attn_weights  # attn_weights: (B, num_heads, T_q, T_ctx)
        return out
