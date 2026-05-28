## 2024-05-15 - MultiheadAttention memory leak prevention
**Learning:** PyTorch `nn.MultiheadAttention` will compute and allocate attention weights by default, which can break optimization backends like FlashAttention.
**Action:** When using `nn.MultiheadAttention`, if the weights are discarded (`_, weights = ...` -> `_`), always pass `need_weights=False` to optimize memory and computation. However, be extremely careful not to pass `need_weights=False` if the weights are being assigned to a named variable for later use (e.g., `cross_attn_weights`).
