## 2024-05-14 - PyTorch MultiheadAttention Optimization
**Learning:** PyTorch's `nn.MultiheadAttention` computes and returns attention weights by default, which takes memory and disables highly optimized attention backends (e.g., FlashAttention).
**Action:** When using `nn.MultiheadAttention` where attention weights are discarded, always pass `need_weights=False`. However, when attention weights are explicitly unpacked and used (e.g. `cross_attn_weights`), we cannot set `need_weights=False`, or it will result in an unpack error because it returns `None`.
