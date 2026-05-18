## 2024-05-18 - [Optimizing PyTorch MultiheadAttention]
**Learning:** Adding `need_weights=False` to `nn.MultiheadAttention` calls in PyTorch provides a significant memory and speed boost by avoiding unnecessary attention weight materialization and allowing for FlashAttention. However, it's critical to ensure the weights aren't unpacked into named variables downstream, or it causes regressions.
**Action:** When adding `need_weights=False`, add an inline comment explaining the optimization and verify that the second returned element is consistently discarded via `_`.
