## 2023-10-27 - [Optimize MultiheadAttention Weight Computation]
**Learning:** By default, PyTorch's `nn.MultiheadAttention` computes and returns attention weights. When these are unused downstream, it causes unnecessary computation, memory allocation, and prevents the use of optimized backends like FlashAttention.
**Action:** Always set `need_weights=False` in `nn.MultiheadAttention` forward calls when the attention weights are not explicitly required for downstream processing.
