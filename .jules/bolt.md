## 2024-05-01 - Optimizing MultiheadAttention

**Learning:** In PyTorch, `nn.MultiheadAttention` computes attention weights by default, which takes memory and computation. When these weights are discarded (`attn_out, _ = self.mha(..., need_weights=False)`), setting `need_weights=False` can skip this unused computation and enables using optimized attention backends (e.g., FlashAttention).
**Action:** When using `nn.MultiheadAttention` and the attention weights (the second return value) are unused, always specify `need_weights=False`.
