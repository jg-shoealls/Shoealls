## 2025-01-20 - MultiheadAttention Memory Optimization
**Learning:** By default, PyTorch's `nn.MultiheadAttention` computes and returns attention weights. If they are unused, it prevents optimized backends (like FlashAttention) from being used and wastes memory/computation.
**Action:** Always check if attention weights are needed downstream. If not, explicitly pass `need_weights=False` and unpack to `_` with a comment explaining the optimization.
