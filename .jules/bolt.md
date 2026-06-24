## 2024-03-24 - PyTorch MultiheadAttention Optimization
**Learning:** When using PyTorch's `nn.MultiheadAttention`, passing `need_weights=False` prevents unnecessary computation and memory allocation if attention weights are discarded, enabling optimized attention backends (like FlashAttention).
**Action:** Always set `need_weights=False` in `nn.MultiheadAttention` calls if the returned attention weights are ignored.
