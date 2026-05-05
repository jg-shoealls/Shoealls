## 2024-05-13 - PyTorch MultiheadAttention Optimization
**Learning:** PyTorch's `nn.MultiheadAttention` computes and returns attention weights by default. If the weights are discarded (`_`), memory and computation are wasted, and optimized attention backends (like FlashAttention) cannot be used.
**Action:** When using `nn.MultiheadAttention`, always set `need_weights=False` unless the attention matrix is explicitly needed. But be careful not to unpack the output if weights are needed elsewhere in the code.
