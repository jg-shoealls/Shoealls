
## 2024-07-18 - [Optimize PyTorch nn.MultiheadAttention by skipping weight computation]
**Learning:** In PyTorch, `nn.MultiheadAttention` computes both the attention output and the attention weights by default. If the weights are returned but not used (e.g., unpacked into `_`), this results in unnecessary computation and memory allocation.
**Action:** Always pass `need_weights=False` to `nn.MultiheadAttention` when the attention weights are not needed downstream. This prevents unnecessary allocations and can enable optimized attention backends like FlashAttention. If the returned weight was being assigned to an unused variable, safely rename it to `_` during unpacking.
