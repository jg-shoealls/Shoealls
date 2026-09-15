## 2024-05-18 - PyTorch MultiheadAttention need_weights Optimization
**Learning:** When using PyTorch's `nn.MultiheadAttention`, if the attention weights are not explicitly needed, setting `need_weights=False` prevents unnecessary computation and memory allocation, and enables optimized backend implementations like FlashAttention.
**Action:** Always set `need_weights=False` in `nn.MultiheadAttention` forward passes when the resulting attention weights are ignored or unpacked but unused.
