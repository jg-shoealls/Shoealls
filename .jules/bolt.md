## 2024-05-18 - [Optimizing PyTorch Attention]
**Learning:** PyTorch `nn.MultiheadAttention` computes attention weights by default, which prevents the use of optimized memory-efficient backends like FlashAttention.
**Action:** When attention weights are not needed (e.g., when the return value is discarded using `_`), explicitly set `need_weights=False` to enable faster computation and reduce memory allocation.
## 2024-05-18 - [API Error Handling Data Structure Check]
**Learning:** `mag_baro` wasn't fully added to schemas or gracefully handled leading to `KeyError`. Also, hardcoded class maps can fail when configurations change.
**Action:** When working on APIs, ensure new inputs/outputs like optional model modalities are specified in both `schemas.py` and `service.py`. Make sure `GAIT_CLASS_NAMES` lengths align with `num_classes` in configs.
## 2024-05-18 - [API Configuration and Missing Test Modules]
**Learning:** Hardcoding `num_classes` sizes in API logic or test assertions is prone to breakage when `default.yaml` is modified or changed (e.g. from 4 to 11 classes). Also, attempting to import tests for missing or uncommitted modules (like `test_disease_biomarkers.py`) causes complete `pytest` collection failure.
**Action:** When validating sizes from models instantiated via config, read the dynamic property directly (e.g., `config["data"]["num_classes"]`). When resolving CI pipelines on large test suites, always test collection using plain `pytest tests` locally and remove/ignore missing modules rather than just trying to run previously targeted subsets.
