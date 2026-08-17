## 2024-08-17 - Unused attention weights in nn.MultiheadAttention
**Learning:** In PyTorch's `nn.MultiheadAttention`, if the attention weights (the second output) are discarded, setting `need_weights=False` prevents unnecessary computation and memory allocation, and enables optimized attention backends like FlashAttention.
**Action:** When using `nn.MultiheadAttention`, if the returned weights are not used or just assigned to `_`, explicitly pass `need_weights=False`. If it was unpacking into a variable that is unused downstream, rename it to `_` to satisfy linting rules and clarify intent.
