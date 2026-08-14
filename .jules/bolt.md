## 2024-05-19 - [Optimize MultiheadAttention to enable FlashAttention]
**Learning:** PyTorch's `nn.MultiheadAttention` calculates and returns attention weights by default. If these weights are unpacked but unused downstream, applying `need_weights=False` safely prevents this unnecessary computation and memory allocation.
**Action:** When using `nn.MultiheadAttention` and explicitly ignoring the returned attention weights, set `need_weights=False` to unlock optimized attention backends like FlashAttention and improve performance.
