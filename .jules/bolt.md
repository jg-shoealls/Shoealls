## 2024-05-27 - [MultiheadAttention need_weights optimization]
**Learning:** PyTorch `nn.MultiheadAttention` allocates memory for attention weights by default. Setting `need_weights=False` avoids this allocation when the weights are not used, improving performance and potentially enabling optimized backends like FlashAttention.
**Action:** When `nn.MultiheadAttention` returns a tuple where the second element (weights) is unused (or explicitly unpacked into a named variable but never used), update the call to include `need_weights=False` and assign the return tuple with a `_` for the weights.
