## 2024-05-18 - PyTorch FlashAttention Optimization
**Learning:** In PyTorch's nn.MultiheadAttention, attention weights are computed by default. If they are unused, setting need_weights=False prevents unnecessary computation and memory allocation, enabling FlashAttention.
**Action:** Add need_weights=False to self_attention and cross_attn calls in fusion.py and rename the unused returned weights to _ with an explanatory comment.
