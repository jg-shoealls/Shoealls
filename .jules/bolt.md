## 2024-05-24 - [PyTorch MultiheadAttention Memory and Compute Overhead]
**Learning:** By default, PyTorch's `nn.MultiheadAttention` computes and returns the attention weights (`attn_weights`). This involves additional memory allocation and computation which can be bypassed (and potentially enable faster attention kernels like FlashAttention) if the weights are discarded anyway.
**Action:** Always set `need_weights=False` in PyTorch's `nn.MultiheadAttention` forward passes when the returned attention weights are ignored. When returning unpacked but unused variables, assign them to `_`.
