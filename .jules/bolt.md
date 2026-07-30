## 2024-05-24 - [PyTorch MultiheadAttention Optimization]
**Learning:** By setting `need_weights=False` on `nn.MultiheadAttention` and replacing unused unpacking variables (like `cross_attn_weights` and `_`) with `_`, we can prevent unnecessary memory allocation and enable optimized backends like FlashAttention.
**Action:** Apply `need_weights=False` to `nn.MultiheadAttention` when attention weights are unused, rename unpack targets to `_`, and add an explanatory comment.
