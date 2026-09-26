## 2024-05-24 - [PyTorch FlashAttention enable]
**Learning:** In PyTorch, calling `nn.MultiheadAttention` with `need_weights=False` prevents unnecessary computation and memory allocation when the attention weights are not used downstream, which also enables FlashAttention.
**Action:** Always set `need_weights=False` in `nn.MultiheadAttention` calls if the weights output is discarded (e.g., using `_`).
