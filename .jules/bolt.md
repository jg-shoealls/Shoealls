
## 2024-05-18 - [Optimize PyTorch nn.MultiheadAttention with need_weights=False]
**Learning:** PyTorch's `nn.MultiheadAttention` allocates memory and computes attention weights by default, which can cause performance bottlenecks when these weights are unused and thrown away (e.g. `_, _ = attn(..., need_weights=True)`).
**Action:** When using `nn.MultiheadAttention`, explicitly set `need_weights=False` if the attention weights are not needed for downstream tasks to prevent unnecessary computations and memory usage. Safely rename the unpacking target to `_`.
