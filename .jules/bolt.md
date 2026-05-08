## 2024-05-08 - [Optimize nn.MultiheadAttention by explicitly omitting weights]
**Learning:** PyTorch's `nn.MultiheadAttention` computes attention weights by default, even if they aren't used, adding overhead and preventing the use of optimized backends like FlashAttention. We can skip this by explicitly passing `need_weights=False`.
**Action:** Always check if the second return value (`attn_weights`) of `nn.MultiheadAttention` is being discarded into `_`. If so, explicitly pass `need_weights=False` to the `forward()` method to save memory and potentially speed up computation.
