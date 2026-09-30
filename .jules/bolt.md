## 2024-09-30 - MultiheadAttention and FlashAttention
**Learning:** PyTorch's `nn.MultiheadAttention` computes and returns attention weights by default, which prevents the use of memory-efficient FlashAttention implementations under the hood and incurs unnecessary memory and computation overhead.
**Action:** When attention weights are discarded or not explicitly needed downstream, always pass `need_weights=False` to the `nn.MultiheadAttention` forward pass to enable FlashAttention. If the weight tensor is unpacked but unused, rename it to `_`.
