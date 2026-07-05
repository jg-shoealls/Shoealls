## 2024-05-24 - [PyTorch MultiheadAttention Memory Optimization]
**Learning:** When using PyTorch's `nn.MultiheadAttention`, if the attention weights are explicitly unpacked into a named variable (e.g., `cross_attn_weights`) but used nowhere else downstream, it allocates unnecessary memory.
**Action:** Always set `need_weights=False` and unpack the weights as `_` when the returned attention weights are not required, to prevent unnecessary computation and memory allocation and enable optimized attention backends (like FlashAttention).
