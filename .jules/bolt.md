
## 2026-08-27 - [Optimize MultiheadAttention Performance]
**Learning:** In PyTorch, `nn.MultiheadAttention` computes and returns attention weights by default, allocating significant memory and compute resources. When these weights are discarded and not used downstream, this is an unnecessary overhead, particularly in multi-modal architectures with many attention layers.
**Action:** Always set `need_weights=False` in the forward pass of `nn.MultiheadAttention` when the returned attention weights are ignored (e.g. `attn_out, _ = self.attn(...)`), allowing for optimized attention backends (like FlashAttention) and memory savings.
