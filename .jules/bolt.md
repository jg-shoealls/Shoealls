## 2024-07-19 - PyTorch MultiheadAttention Optimization
**Learning:** PyTorch's `nn.MultiheadAttention` computes attention weights by default even if they are immediately discarded (e.g., using `_`), consuming unnecessary memory and computation, preventing optimized backends like FlashAttention from being fully utilized.
**Action:** When using `nn.MultiheadAttention`, explicitly pass `need_weights=False` if the attention weights return tensor is not used downstream. Unpack the second return value into `_` to satisfy linting.

## 2024-07-19 - Dictionary KeyError Prevention
**Learning:** `MultimodalGaitNet` unconditionally accesses the `mag_baro` key from the parsed `batch` dictionary, even when `SensorData` schemas omit it. This omission raises a `KeyError: 'mag_baro'`.
**Action:** When extracting nested attributes to pass to downstream models that assume they exist, inject dummy tensors filled with zeroes (e.g. `torch.zeros((1, mag_baro_channels, seq_len))`) into the `batch` dictionary before the forward pass.

## 2024-07-19 - PyArrow & NumPy ABI Compatibility
**Learning:** `pyarrow` (often a dependency of `datasets` or `pandas`) has ABI compatibility issues with `numpy>=2.0`. When `numpy>=2.0` is installed alongside an older `pyarrow` version, attempting to import or use it triggers `AttributeError: _ARRAY_API not found` or `ImportError: numpy.core.multiarray failed to import` and hard crashes the process (e.g. during FastAPI startup in Docker).
**Action:** In environments relying on `pyarrow` or `datasets`, explicitly pin numpy to `<2` (e.g., `numpy>=1.24.0,<2`) in `requirements.txt` to prevent pip from resolving to `numpy 2.x`.
