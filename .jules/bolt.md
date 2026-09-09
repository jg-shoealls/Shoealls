
## 2024-05-18 - [Optimization of Attention Mechanism]
**Learning:** In models using PyTorch's `nn.MultiheadAttention`, if attention weights are not used (e.g. they are ignored or dropped with `_`), failing to set `need_weights=False` forces unnecessary calculations, increasing memory and compute time.
**Action:** Always check the return values of `nn.MultiheadAttention` and if the attention matrix is not required, explicitly set `need_weights=False` to trigger optimized attention paths like FlashAttention.
