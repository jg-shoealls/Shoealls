
## 2025-02-24 - PyTorch MultiheadAttention Optimization
**Learning:** By default, `nn.MultiheadAttention` computes and returns attention weights. If these weights are discarded (e.g., using `_`), omitting `need_weights=False` results in unnecessary computation, memory allocation, and prevents PyTorch from using highly optimized attention backends like FlashAttention. However, we must ensure we don't apply this when weights are explicitly unpacked (e.g., `cross_attn_weights`).
**Action:** Always set `need_weights=False` when calling `nn.MultiheadAttention` if the attention weights are not needed for downstream tasks, but verify first that they are indeed ignored.
