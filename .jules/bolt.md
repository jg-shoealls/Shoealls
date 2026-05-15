## 2024-05-18 - [Optimizing PyTorch Attention]
**Learning:** PyTorch `nn.MultiheadAttention` computes attention weights by default, which prevents the use of optimized memory-efficient backends like FlashAttention.
**Action:** When attention weights are not needed (e.g., when the return value is discarded using `_`), explicitly set `need_weights=False` to enable faster computation and reduce memory allocation.
