## 2024-05-24 - [PyTorch nn.MultiheadAttention Optimization]
**Learning:** Adding `need_weights=False` to `nn.MultiheadAttention` calls significantly reduces compute and memory overhead (observed drop from ~1.9s to ~1.2s for 100 iterations on CPU) when attention weights are discarded (assigned to `_`). However, care must be taken not to apply this optimization if the weights are unpacked into named variables, as returning `None` can cause unexpected regressions down the line.
**Action:** Always check how the second return value of `nn.MultiheadAttention` is used. If it's explicitly ignored with `_`, append `need_weights=False`.

## 2024-05-24 - [API mag_baro KeyError in MultimodalGaitNet]
**Learning:** A `KeyError: 'mag_baro'` inside `src/models/multimodal_gait_net.py` was causing CI failures because the API and tests did not supply the `mag_baro` feature but supplied `skeleton` instead. This was because `mag_baro_encoder` was incorrectly put where `skeleton_encoder` was expected in `MultimodalGaitNet`.
**Action:** Replaced `mag_baro_encoder` with `skeleton_encoder` in `MultimodalGaitNet` to align with the expected 3 modalities (IMU, pressure, skeleton). Also updated the `GAIT_CLASS_NAMES` mappings to handle 11 target classes as specified in the configurations.
