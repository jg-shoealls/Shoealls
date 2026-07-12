## 2024-05-24 - PyTorch MultiheadAttention Optimization with Comments
**Learning:** Adding `need_weights=False` to `nn.MultiheadAttention` is highly effective for PyTorch 2.0+ scaled dot-product attention optimization, but it MUST be accompanied by explanatory comments in the codebase to satisfy constraints.
**Action:** When implementing PyTorch model optimizations, explicitly document the change with an inline comment (e.g., `# ⚡ Bolt: need_weights=False to save memory...`) directly above the modified line.
