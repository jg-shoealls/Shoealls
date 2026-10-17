## 2024-10-17 - [FlashAttention Optimization in Fusion]
**Learning:** PyTorch's MultiheadAttention allocates memory and computes attention weights by default. When weights are unpacked to '_' but unused, this wastes computation and prevents FlashAttention from being enabled.
**Action:** Always set need_weights=False in nn.MultiheadAttention forward calls if the attention weights are not needed downstream.
