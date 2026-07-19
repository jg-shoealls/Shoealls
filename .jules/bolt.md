## 2024-07-19 - PyTorch MultiheadAttention Optimization
**Learning:** PyTorch's `nn.MultiheadAttention` computes attention weights by default even if they are immediately discarded (e.g., using `_`), consuming unnecessary memory and computation, preventing optimized backends like FlashAttention from being fully utilized.
**Action:** When using `nn.MultiheadAttention`, explicitly pass `need_weights=False` if the attention weights return tensor is not used downstream. Unpack the second return value into `_` to satisfy linting.

## 2024-07-19 - Dictionary KeyError Prevention
**Learning:** `MultimodalGaitNet` unconditionally accesses the `mag_baro` key from the parsed `batch` dictionary, even when `SensorData` schemas omit it. This omission raises a `KeyError: 'mag_baro'`.
**Action:** When extracting nested attributes to pass to downstream models that assume they exist, inject dummy tensors filled with zeroes (e.g. `torch.zeros((1, mag_baro_channels, seq_len))`) into the `batch` dictionary before the forward pass.
