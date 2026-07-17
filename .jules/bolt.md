## 2024-05-24 - [Optimize MultiheadAttention with need_weights=False]
**Learning:** PyTorch's `nn.MultiheadAttention` computes attention weights by default even when they are discarded, causing unnecessary memory allocation and preventing the use of optimized backends like FlashAttention.
**Action:** When unused, set `need_weights=False` on `MultiheadAttention` calls and rename unpacked target to `_`.
