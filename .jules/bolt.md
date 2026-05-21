## 2024-05-21 - [Optimize nn.MultiheadAttention with need_weights=False]
**Learning:** PyTorch's `nn.MultiheadAttention` computes attention weights by default, which takes extra memory and time. We can pass `need_weights=False` to enable faster backends like FlashAttention, unless the output attention weights are actually unpacked and used.
**Action:** When using `nn.MultiheadAttention(..., need_weights=False)`, make sure to replace `attn_out, _ = self.attn(...)` or check if the returned weights are used elsewhere before changing it.
