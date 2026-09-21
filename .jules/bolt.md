## 2026-09-21 - [MultiheadAttention need_weights Optimization]
**Learning:** [When using PyTorch's nn.MultiheadAttention, setting need_weights=False prevents unnecessary computation and memory allocation if attention weights are discarded, which enables FlashAttention.]
**Action:** [Set need_weights=False when calling nn.MultiheadAttention if the attention weights are not used downstream.]
