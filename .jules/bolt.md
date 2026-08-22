
## 2024-05-18 - [Optimize MultiheadAttention with need_weights=False]
**Learning:** PyTorch `nn.MultiheadAttention` calculates attention weights by default even when they are immediately discarded by unpacking into `_` or unused named variables like `cross_attn_weights`.
**Action:** Always set `need_weights=False` in the forward pass of `MultiheadAttention` (e.g., `self.cross_attn(..., need_weights=False)`) when attention weights are unused downstream, preventing unnecessary computation and enabling memory-efficient backends like FlashAttention.

## 2024-05-18 - [Docker Healthchecks & Numpy Constraint]
**Learning:** The ML models are heavy and application startup is slow, leading to Docker container premature termination via healthcheck. Furthermore, the test suite may break due to missing array API from older numpy versions interacting with other deps (like pyarrow/scipy).
**Action:** Extend Docker healthchecks initialization (e.g., `start_period=60s`, `timeout=30s`, `retries=5`) to prevent premature termination, and correctly pin numpy to `>=1.26.4,<2.0.0` in `requirements.txt` to avoid crashes related to array_api and pyarrow.
