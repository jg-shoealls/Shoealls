## 2025-02-28 - [Performance] Optimize MultiheadAttention
**Learning:** PyTorch's `nn.MultiheadAttention` calculates and returns attention weights by default. If the downstream code discards these weights, memory allocation and computation are wasted, preventing optimized backend paths like FlashAttention.
**Action:** Always set `need_weights=False` on `nn.MultiheadAttention` forward passes if the attention weights are unused, unpacking to `_` when appropriate.
