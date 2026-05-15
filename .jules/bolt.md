## 2024-05-18 - [Optimizing PyTorch Attention]
**Learning:** PyTorch `nn.MultiheadAttention` computes attention weights by default, which prevents the use of optimized memory-efficient backends like FlashAttention.
**Action:** When attention weights are not needed (e.g., when the return value is discarded using `_`), explicitly set `need_weights=False` to enable faster computation and reduce memory allocation.
## 2024-05-18 - [API Error Handling Data Structure Check]
**Learning:** `mag_baro` wasn't fully added to schemas or gracefully handled leading to `KeyError`. Also, hardcoded class maps can fail when configurations change.
**Action:** When working on APIs, ensure new inputs/outputs like optional model modalities are specified in both `schemas.py` and `service.py`. Make sure `GAIT_CLASS_NAMES` lengths align with `num_classes` in configs.
