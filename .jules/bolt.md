## 2024-10-10 - Disable unused MultiheadAttention weights for FlashAttention
**Learning:** PyTorch's `nn.MultiheadAttention` computes and returns attention weights by default, which blocks the use of fast path (FlashAttention) and wastes memory/compute when these weights are unused (e.g., in `src/models/fusion.py`).
**Action:** Always set `need_weights=False` when calling `nn.MultiheadAttention` if the attention weights are just unpacked and discarded (e.g., `attn_out, _ = self.attn(..., need_weights=False)`). Ensure downstream components do not actually rely on them before applying.
