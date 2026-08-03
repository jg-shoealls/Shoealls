
## 2024-08-03 - nn.MultiheadAttention optimization
**Learning:** In PyTorch, calling `nn.MultiheadAttention` with the default `need_weights=True` calculates and returns attention weights. This prevents using optimized attention implementations (like FlashAttention) which are often fused and do not materialize the attention matrix. If the returned attention weights are discarded or unused downstream, setting `need_weights=False` avoids allocating memory for this large matrix and speeds up both the forward and backward pass significantly.
**Action:** Always check if the second return value (`attn_weights`) of `nn.MultiheadAttention` is actually used. If it's unused or simply assigned to `_`, explicitly pass `need_weights=False` to the forward call to enable FlashAttention and save memory.
