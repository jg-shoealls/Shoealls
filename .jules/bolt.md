## 2024-05-18 - [Optimizing `nn.MultiheadAttention`]
**Learning:** PyTorch's `nn.MultiheadAttention` computes and returns attention weights by default, which introduces unnecessary computation and memory allocation if those weights are discarded (e.g., using `_`).
**Action:** When using `nn.MultiheadAttention` and the attention weights are not needed, set `need_weights=False` in the forward pass call to enable optimized attention backends (like FlashAttention) and improve performance.
