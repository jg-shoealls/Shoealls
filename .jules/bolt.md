## 2024-05-24 - [PyTorch nn.MultiheadAttention Optimization]
**Learning:** PyTorch's `nn.MultiheadAttention` computes and returns attention weights by default, which consumes unnecessary memory and prevents the use of optimized attention backends (like FlashAttention).
**Action:** When the attention weights from `nn.MultiheadAttention` are not explicitly used downstream, always set `need_weights=False` and assign the second return value to `_`.
