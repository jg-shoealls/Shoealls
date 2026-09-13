## 2026-09-13 - PyTorch MultiheadAttention Memory & Computation Optimization
**Learning:** Setting need_weights=False in PyTorch's nn.MultiheadAttention prevents unnecessary computation and memory allocation when attention weights are discarded, enabling optimized attention backends (like FlashAttention).
**Action:** Always pass need_weights=False to nn.MultiheadAttention when the attention weights are not needed downstream.
