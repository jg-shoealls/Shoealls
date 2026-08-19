## 2025-02-19 - [Optimize nn.MultiheadAttention with need_weights=False]
**Learning:** PyTorch's `nn.MultiheadAttention` returns both attention output and weights by default. Returning weights prevents the use of optimized attention backends (e.g., FlashAttention), causing unnecessary computation and memory allocation when the weights are not used downstream.
**Action:** When `nn.MultiheadAttention` attention weights are not needed, explicitly set `need_weights=False` and safely rename the unpacking target to `_`.
