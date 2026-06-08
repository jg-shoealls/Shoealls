## 2024-06-08 - PyTorch MultiheadAttention Optimization
**Learning:** PyTorch's `nn.MultiheadAttention` computes attention weights by default, which takes memory and computation, and prevents the use of optimized fast paths like FlashAttention. We can disable this when the weights are ignored by adding `need_weights=False`.
**Action:** Always verify if attention weights are unpacked into ignored variables (`_`). If so, add `need_weights=False` to the MultiheadAttention forward call.
