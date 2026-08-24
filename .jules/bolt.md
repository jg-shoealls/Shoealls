## 2023-10-25 - [Optimize MultiheadAttention with need_weights=False]
**Learning:** PyTorch's `nn.MultiheadAttention` returns attention weights by default, which disables optimized attention backends like FlashAttention and increases memory allocation.
**Action:** Always set `need_weights=False` when the returned attention weights are not explicitly used downstream.
