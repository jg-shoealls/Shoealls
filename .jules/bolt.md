## 2024-05-18 - [PyTorch MultiheadAttention Memory Opt]
**Learning:** Setting `need_weights=False` in PyTorch's `nn.MultiheadAttention` prevents memory allocation for attention weight matrices, enabling memory-efficient and faster attention computation (like FlashAttention backend).
**Action:** Always check if attention weights are unpacked and used downstream; if not, pass `need_weights=False` and assign the weight output to `_`.
