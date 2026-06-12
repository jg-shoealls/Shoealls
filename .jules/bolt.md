## 2024-05-18 - [Optimize nn.MultiheadAttention memory footprint]
**Learning:** Discarding attention weights unpacked as `_` causes unnecessary memory allocation and blocks FlashAttention optimizations.
**Action:** Always set `need_weights=False` in PyTorch's `nn.MultiheadAttention` when the weights are not explicitly needed.
