## 2026-07-14 - [PyTorch Attention Optimization]
**Learning:** Passing `need_weights=False` to `nn.MultiheadAttention` in PyTorch avoids allocating memory for the attention weights, which unlocks the use of fast attention backends (like FlashAttention).
**Action:** When using `nn.MultiheadAttention`, always check if the returned attention weights are actually used. If they are discarded (e.g., unpacked as `_`), add `need_weights=False` to the forward call to improve performance.
