## 2024-05-24 - PyTorch MultiheadAttention Optimization with Comments
**Learning:** Adding `need_weights=False` to `nn.MultiheadAttention` is highly effective for PyTorch 2.0+ scaled dot-product attention optimization, but it MUST be accompanied by explanatory comments in the codebase to satisfy constraints.
**Action:** When implementing PyTorch model optimizations, explicitly document the change with an inline comment (e.g., `# ⚡ Bolt: need_weights=False to save memory...`) directly above the modified line.

## 2024-05-24 - PyTorch MultiheadAttention Optimization with Comments
**Learning:** Adding `need_weights=False` to `nn.MultiheadAttention` is highly effective for PyTorch 2.0+ scaled dot-product attention optimization, but it MUST be accompanied by explanatory comments in the codebase to satisfy constraints.
**Action:** When implementing PyTorch model optimizations, explicitly document the change with an inline comment (e.g., `# ⚡ Bolt: need_weights=False to save memory...`) directly above the modified line.

## 2024-05-24 - Multi-class classification index out of bounds error
**Learning:** During prediction or explanation, we were mapping model output indices to names directly via list index or dict index. However, the number of target classes (11) was much larger than the default mapped names (4) in `api/service.py` (`GAIT_CLASS_NAMES`), causing `KeyError` or `IndexError` during tests where the prediction class exceeded index 3.
**Action:** Always use `.get()` with dictionary fallback (or boundary checks for arrays) to ensure missing/undefined indices are handled gracefully, preventing 500 API errors when config class number overrides class names.

## 2024-05-24 - Numpy/PyArrow 2.0 Incompatibility
**Learning:** `pyarrow==14.0.2` fails to import when the environment builds against Numpy 2.x+, causing fatal 500 errors or CI failures because its internal C extension expects Numpy 1.x `_ARRAY_API`.
**Action:** When updating packages or dealing with environment setups in this codebase, actively constrain Numpy to `<2.0.0` in `requirements.txt` (`numpy>=1.24.0,<2.0.0`) to avoid PyArrow dependency crashing at runtime.
