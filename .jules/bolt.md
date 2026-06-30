## 2024-06-25 - [Memory Instructions for need_weights]
**Learning:** PyTorch's `nn.MultiheadAttention` calculates attention weights by default, but these can be disabled with `need_weights=False` if not needed, potentially enabling FlashAttention and reducing memory usage.
**Action:** When `need_weights=False` can be safely used without altering behavior (if attention weights aren't used), apply it. If they are unpacked into a named variable but unused, rename it to `_` first.
