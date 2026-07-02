## 2024-05-24 - [PyTorch FlashAttention Optimization]
**Learning:** [In PyTorch, setting `need_weights=False` on `nn.MultiheadAttention` enables FlashAttention memory and computation optimizations when attention weight maps are unnecessary and immediately discarded.]
**Action:** [Check all attention unpackings in model forward passes and set `need_weights=False` if weights are assigned to `_` or safely rename unused variables to `_`.]
