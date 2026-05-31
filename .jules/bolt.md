## 2024-05-31 - [FlashAttention Enablement]
**Learning:** Using `nn.MultiheadAttention` without `need_weights=False` forces the computation and allocation of attention weights even if they are discarded, blocking optimized backends like FlashAttention.
**Action:** Always set `need_weights=False` when attention weights are discarded into `_`. Ensure we don't apply this when weights are explicitly named and unpacked.
