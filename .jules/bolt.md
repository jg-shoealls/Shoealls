## 2024-05-24 - PyTorch MultiheadAttention Optimization
**Learning:** When using PyTorch's nn.MultiheadAttention, leaving need_weights=True (the default) causes unnecessary computation and memory allocation even if the weights are discarded (e.g., using '_'). Setting it to False can enable optimized attention backends like FlashAttention.
**Action:** Always set need_weights=False when calling nn.MultiheadAttention if the attention weights are not needed for downstream usage.
