
## 2026-09-11 - [MultiheadAttention Optimization]
**Learning:** In PyTorch's nn.MultiheadAttention, attention weights are computed and returned by default even if unpacked into _. This prevents optimized backends like FlashAttention from engaging and wastes memory.
**Action:** Always set need_weights=False when calling nn.MultiheadAttention if the weights are not explicitly used downstream.
