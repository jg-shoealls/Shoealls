## 2024-08-06 - [PyTorch nn.MultiheadAttention memory optimization]
**Learning:** PyTorch's `nn.MultiheadAttention` returns both the attention output and the attention weights by default. If the attention weights are not used downstream, this results in unnecessary memory allocation and compute.
**Action:** Always set `need_weights=False` when calling `nn.MultiheadAttention` forward passes if the attention weights are discarded. If unpacked into a named variable, safely rename the target to `_`.
