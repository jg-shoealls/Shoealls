## 2026-06-21 - Optimize PyTorch MultiheadAttention for FlashAttention
**Learning:** In PyTorch's `nn.MultiheadAttention`, the default behavior returns attention weights, which prevents the use of optimized memory-efficient attention backends (like FlashAttention). If attention weights are not used downstream, this wastes compute and memory.
**Action:** When using `nn.MultiheadAttention` where attention weights are discarded (e.g., assigned to `_`), explicitly set `need_weights=False` in the forward call to enable optimized attention backends.
