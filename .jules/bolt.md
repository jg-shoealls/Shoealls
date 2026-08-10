## 2024-08-10 - [PyTorch MultiheadAttention Memory & Computation Optimization]
**Learning:** PyTorch `nn.MultiheadAttention` computes attention weights by default, which can cause unnecessary computation and memory allocation if the weights are discarded later. Unpacking into a variable that isn't used down the line (e.g. `cross_attn_weights`) is a red flag for this issue.
**Action:** Always set `need_weights=False` in `nn.MultiheadAttention` calls if the returned attention weights are not needed, and rename the unpacking target to `_`.
