## 2024-05-15 - [PyTorch Attention Optimization]
**Learning:** PyTorch's `nn.MultiheadAttention` computes attention weights by default even if they are not used downstream, which allocates unnecessary memory and prevents optimizations like FlashAttention.
**Action:** When using `nn.MultiheadAttention` and discarding the second return value (the attention weights), explicitly pass `need_weights=False` in the forward pass.
