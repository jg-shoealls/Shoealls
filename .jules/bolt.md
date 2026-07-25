## 2024-05-18 - Optimize MultiheadAttention weights
**Learning:** PyTorch's MultiheadAttention computes and returns attention weights by default, which takes memory and computation.
**Action:** Set need_weights=False when unpacking into an unused variable, and explicitly rename it to _ to safely discard it.
