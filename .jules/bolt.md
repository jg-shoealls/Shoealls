## 2024-05-18 - Optimize PyTorch MultiheadAttention Memory and Speed
**Learning:** PyTorch `nn.MultiheadAttention` computes attention weights by default, which takes significant memory and prevents the use of optimized fast paths like FlashAttention when `need_weights=True`.
**Action:** When using `nn.MultiheadAttention` where the attention weights return tuple is unused (i.e. unpacked to `_`), set `need_weights=False` to skip the computation and enable fast paths. Be careful not to apply this if the attention weights are explicitly unpacked and used.
