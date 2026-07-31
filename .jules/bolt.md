## 2025-07-31 - Optimization of PyTorch MultiheadAttention
**Learning:** PyTorch's `nn.MultiheadAttention` supports a `need_weights` flag. When attention weights are not needed for downstream tasks (e.g., they are unpacked to an unused variable), setting `need_weights=False` prevents unnecessary computation and memory allocation. It also enables optimized attention backends like FlashAttention, significantly boosting performance.
**Action:** When using `nn.MultiheadAttention` where the attention weights output is unused, explicitly set `need_weights=False` in the forward pass.
