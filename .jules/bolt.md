## 2024-05-24 - PyTorch MultiheadAttention Memory Optimization
**Learning:** PyTorch's `nn.MultiheadAttention` computes and returns attention weights by default, which can cause unnecessary computation and memory allocation. In this project, when these weights are unused and unpacked to `_`, FlashAttention cannot be leveraged.
**Action:** Always set `need_weights=False` in `nn.MultiheadAttention` forward calls when the attention weights are explicitly unpacked and unused downstream.
