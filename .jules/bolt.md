## 2024-05-24 - [Optimize MultiheadAttention with need_weights=False]
**Learning:** PyTorch `nn.MultiheadAttention` computes attention weights by default, which can be memory and compute-intensive. If the returned attention weights are completely unused, setting `need_weights=False` is a significant optimization that allows the engine to skip the calculation entirely or use more optimized paths (e.g., FlashAttention).
**Action:** Always check if the second return value (attention weights) of `nn.MultiheadAttention` is actually used. If it is ignored or discarded (e.g., using `_`), explicitly set `need_weights=False` to optimize performance.
## 2024-05-24 - [Fix CI Failures related to Dictionary Get and Model Inputs]
**Learning:**
1. When dictionaries like `GAIT_CLASS_NAMES` or `CLASS_NAMES_KR` only have subsets of expected index mappings (e.g. 0-3) while the model output predicts larger indices due to configuration mismatches (`num_classes: 11` instead of `4`), a simple `.get(idx)` with a fallback is insufficient if you use list syntax instead of dictionary mapping. In Python, lists do not have a `.get()` method. Use array boundary checks or ensure mapping structures are dictionaries before using `.get()`.
2. When creating models or adding features that expect certain input tensors (e.g., `mag_baro`), make sure the input data pipelines (like `api/service.py` `_sensor_to_tensors`, dummy datasets in tests) explicitly inject those tensors to avoid `KeyError` during the forward pass.

**Action:**
- Ensure list access is guarded with `idx < len(my_list)` rather than assuming a list has `.get()`.
- Explicitly populate missing expected tensors like `mag_baro` in the data ingestion step if they are missing from external requests but required by the unified multimodal model.
## 2024-05-24 - [Fix Docker Healthcheck Failures due to Slow Model Loading]
**Learning:** Due to heavy ML model loading in memory, the API server can take longer to start than standard web applications. Short Docker healthchecks (like `start_period=15s`, `timeout=10s`) will fail, leading to premature container termination before the app is fully ready.
**Action:** When updating or debugging Docker setups in this repository, ensure healthcheck definitions (in `Dockerfile` or `docker-compose.yml`) use extended initialization parameters (`start_period=60s`, `timeout=30s`, `retries=5`) to prevent CI and deployment pipeline failures.
