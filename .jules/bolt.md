
## 2024-05-18 - [Optimize MultiheadAttention with need_weights=False]
**Learning:** PyTorch `nn.MultiheadAttention` calculates attention weights by default even when they are immediately discarded by unpacking into `_` or unused named variables like `cross_attn_weights`.
**Action:** Always set `need_weights=False` in the forward pass of `MultiheadAttention` (e.g., `self.cross_attn(..., need_weights=False)`) when attention weights are unused downstream, preventing unnecessary computation and enabling memory-efficient backends like FlashAttention.
