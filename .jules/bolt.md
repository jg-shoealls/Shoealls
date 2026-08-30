## 2024-10-24 - [PyTorch MultiheadAttention Optimization]
**Learning:** In PyTorch's `nn.MultiheadAttention`, explicitly passing `need_weights=False` prevents unnecessary computation and memory allocation if attention weights are not needed, which also enables optimized attention backends like FlashAttention.
**Action:** When using `nn.MultiheadAttention`, always check if the second return value (attention weights) is actually used downstream. If it is only unpacked and discarded, append `need_weights=False` to the forward call.
