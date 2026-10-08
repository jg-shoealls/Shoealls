## 2024-05-18 - [PyTorch FlashAttention Optimization]
**Learning:** PyTorch `nn.MultiheadAttention` computes attention weights by default even if they are immediately discarded (as the variable `_`), preventing the use of optimized FlashAttention kernels and consuming unnecessary memory/compute.
**Action:** Always set `need_weights=False` when calling `nn.MultiheadAttention` if the attention weights are not explicitly used downstream, to automatically enable fast path executions (FlashAttention).
