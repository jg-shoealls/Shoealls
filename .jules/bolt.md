## 2024-05-24 - [PyTorch FlashAttention Optimization]
**Learning:** PyTorch's `nn.MultiheadAttention` calculates and returns attention weights by default, which wastes memory and computation, and disables memory-efficient FlashAttention if the weights are discarded immediately.
**Action:** Always set `need_weights=False` when calling `nn.MultiheadAttention` if the attention weights are not explicitly needed downstream.
