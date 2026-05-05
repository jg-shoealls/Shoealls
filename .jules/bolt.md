## 2024-05-18 - MultiheadAttention memory and compute optimization
**Learning:** By default `nn.MultiheadAttention` calculates and returns attention weights (`attn_weights`). This computation is unnecessary and consumes extra memory when the weights are not used, and it blocks optimized attention backends like FlashAttention from being used.
**Action:** When initializing/calling `nn.MultiheadAttention`, if the attention weights (`attn_weights`) are discarded (using `_`), set `need_weights=False` to enable PyTorch's optimized attention implementations and reduce memory/compute overhead.
