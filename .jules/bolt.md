## 2024-05-30 - [Optimize MultiheadAttention without need_weights]
**Learning:** Using `need_weights=False` in `nn.MultiheadAttention` saves memory and computation when weights are unused, avoiding memory overhead in multi-modal fusion.
**Action:** Always set `need_weights=False` when attention weights are discarded (`_, _ = attn(...)`) to enable optimized backends like FlashAttention.
