## 2025-02-23 - [Optimizing MultiheadAttention memory usage]
**Learning:** PyTorch's `nn.MultiheadAttention` calculates and returns attention weights by default, which can cause significant memory allocation and computation overhead even if the weights are unused, preventing the use of optimized backends like FlashAttention.
**Action:** When using `nn.MultiheadAttention` and the returned attention weights are discarded (e.g., using `_`), explicitly set `need_weights=False` to optimize performance and save memory.
