## 2024-04-25 - MultiheadAttention Memory and Computation Optimization
**Learning:** PyTorch's `nn.MultiheadAttention` computes and returns attention weights by default (`need_weights=True`), which consumes additional memory and prevents the use of highly optimized attention backends like FlashAttention.
**Action:** When using `nn.MultiheadAttention`, if the returned attention weights are discarded (unpacked into `_`), always set `need_weights=False` in the `.forward()` call to save memory and significantly speed up execution via optimized backends.
