## 2024-05-18 - [Optimized MultiheadAttention with FlashAttention]
**Learning:** Unused MultiheadAttention weight unpacks can cause unnecessary computation when FlashAttention is available.
**Action:** Set `need_weights=False` when `cross_attn_weights` (or `attn_out`) are ignored.
