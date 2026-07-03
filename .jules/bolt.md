## 2024-07-03 - [PyTorch MultiheadAttention Optimization]
**Learning:** By default, PyTorch's `nn.MultiheadAttention` returns both the attention output and attention weights. If the attention weights are not explicitly needed downstream, computing and returning them unnecessarily allocates memory and prevents the use of optimized attention backends (such as FlashAttention).
**Action:** When unpacking `MultiheadAttention` returns, always check if the weight tensor is discarded or unused. If so, add `need_weights=False` to the call to optimize performance and enable FlashAttention.
