## 2024-05-18 - [MultiheadAttention optimization]
**Learning:** PyTorch MultiheadAttention computes attention weights by default, but we don't need them in fusion.py and reasoning_engine.py.
**Action:** Set need_weights=False in MultiheadAttention calls to prevent unnecessary computation and memory allocation.
