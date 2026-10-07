## 2024-05-18 - [Optimization of PyTorch MultiheadAttention]
**Learning:** PyTorch `nn.MultiheadAttention` calculates attention weights by default. If these weights are discarded and unused downstream, the memory allocated for them is unnecessary.
**Action:** When using `nn.MultiheadAttention`, if attention weights aren't explicitly used downstream, append `need_weights=False` to the call to prevent unnecessary computation and memory allocation (which also enables FlashAttention).
