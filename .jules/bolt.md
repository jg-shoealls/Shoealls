## 2025-02-28 - PyTorch MultiheadAttention Optimization
**Learning:** PyTorch's `nn.MultiheadAttention` computes and returns attention weights by default, which is computationally expensive and memory-intensive. When these weights are unused (e.g., `attn_out, _ = ...`), this creates an unnecessary performance overhead.
**Action:** Always explicitly set `need_weights=False` when calling `nn.MultiheadAttention` if the attention weights are explicitly discarded. This not only avoids unnecessary computations but enables PyTorch to use optimized memory-efficient backends like FlashAttention.
