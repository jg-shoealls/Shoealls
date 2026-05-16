## 2024-05-16 - [FlashAttention optimization for nn.MultiheadAttention]
**Learning:** PyTorch`s `nn.MultiheadAttention` will needlessly allocate memory and compute attention weights if `need_weights=True` (the default), which prevents optimized backends like FlashAttention from being utilized.
**Action:** Always set `need_weights=False` when calling `nn.MultiheadAttention` forward passes if the attention weights are not needed (which is typical for simple self/cross-attention blocks).
