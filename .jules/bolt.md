## 2024-08-05 - [PyTorch MultiheadAttention Optimization]
**Learning:** In PyTorch's `nn.MultiheadAttention`, if `need_weights=False` is not set when calling the forward pass, unnecessary computation and memory allocation occur even if the attention weights are discarded.
**Action:** Always set `need_weights=False` (e.g. `attn_out, _ = self.attn(q, k, v, need_weights=False)`) when attention weights are not needed downstream, as it saves memory and may enable optimized backends like FlashAttention.
