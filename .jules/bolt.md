## 2024-07-19 - PyTorch MultiheadAttention Optimization
**Learning:** PyTorch's `nn.MultiheadAttention` computes attention weights by default even if they are immediately discarded (e.g., using `_`), consuming unnecessary memory and computation, preventing optimized backends like FlashAttention from being fully utilized.
**Action:** When using `nn.MultiheadAttention`, explicitly pass `need_weights=False` if the attention weights return tensor is not used downstream. Unpack the second return value into `_` to satisfy linting.
