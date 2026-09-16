## 2026-09-16 - [PyTorch FlashAttention via need_weights]
**Learning:** In PyTorch's `nn.MultiheadAttention`, computing attention weights requires additional memory and computation. If these weights are unpacked but ignored (e.g., `attn_out, _ = self.attn(...)`), the overhead is wasted. Setting `need_weights=False` prevents this and can enable FlashAttention.
**Action:** When using `nn.MultiheadAttention`, always verify if the returned attention weights are actually used. If they are discarded, set `need_weights=False` explicitly to improve performance.
