## 2026-06-25 - Optimize MultiheadAttention memory usage
**Learning:** PyTorch MultiheadAttention computes and returns an attention weights matrix by default, which consumes memory unnecessarily if unused.
**Action:** Always set `need_weights=False` when calling `nn.MultiheadAttention` if the resulting weights tuple is discarded (e.g. `out, _ = self.attn(...)`).
