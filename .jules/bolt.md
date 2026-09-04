## 2026-09-04 - Optimize PyTorch MultiheadAttention
**Learning:** Setting `need_weights=False` in PyTorch's `nn.MultiheadAttention` saves memory and compute by skipping attention weight computation, enabling optimized backends like FlashAttention.
**Action:** Always set `need_weights=False` when attention weights are discarded or completely unused downstream.
