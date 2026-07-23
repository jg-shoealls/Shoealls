## 2023-10-24 - PyTorch MultiheadAttention Optimization
**Learning:** PyTorch's `nn.MultiheadAttention` returns attention weights by default, which can cause unnecessary memory allocation and computation if those weights are discarded.
**Action:** When using `nn.MultiheadAttention` and the returned attention weights are ignored, always set `need_weights=False` to optimize memory and computation.
