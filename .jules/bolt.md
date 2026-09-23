## 2024-05-24 - PyTorch MultiheadAttention Optimization
**Learning:** By default, PyTorch's `nn.MultiheadAttention` computes and returns attention weights, which can be computationally expensive and prevents the use of optimized memory-efficient attention backends (like FlashAttention).
**Action:** When calling `nn.MultiheadAttention` where the attention weights are not explicitly needed downstream (e.g., they are discarded or unpacked into `_`), explicitly set `need_weights=False` to prevent unnecessary computation and memory allocation. Add an explanatory comment when doing so.
