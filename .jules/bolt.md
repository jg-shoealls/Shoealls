## 2024-05-24 - Optimize PyTorch MultiheadAttention Memory and Compute
**Learning:** PyTorch's `nn.MultiheadAttention` calculates and materializes the attention weights matrix by default, returning it alongside the output. In many parts of this codebase's architecture (like `CrossModalAttentionFusion` and `CrossModalEvidenceCollector`), these weights were immediately discarded.
**Action:** Always set `need_weights=False` when calling `nn.MultiheadAttention` if the attention weights are not explicitly required for downstream logic. This avoids unnecessary computation and memory allocation.
