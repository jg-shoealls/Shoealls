## 2024-09-19 - [Optimize MultiheadAttention by disabling weight calculation]
**Learning:** PyTorch's `nn.MultiheadAttention` computes and returns attention weights by default. If these weights are not used (e.g., they are unpacked and discarded with `_`), setting `need_weights=False` prevents unnecessary computation and memory allocation, and can enable optimized backends like FlashAttention.
**Action:** When using `nn.MultiheadAttention`, if attention weights are discarded, pass `need_weights=False` to the forward call and change the target to `_` with an explanatory comment.
