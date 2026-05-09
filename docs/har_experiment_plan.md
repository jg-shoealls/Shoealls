# ShoeAlls HAR/IMU Experiment Plan

This plan turns the Hugging Face search results into a reproducible training path.

## 1. PAMAP2 Baseline

PAMAP2 is the first baseline because it has 52 channels, 100 Hz windows, 12 classes,
and multiple body-worn IMUs. The loader downloads `*_X.npy` and `*_y.npy` directly
from Hugging Face to avoid deprecated dataset scripts.

```powershell
python scripts/train_har_baseline.py `
  --source hf `
  --hf-repo monster-monash/PAMAP2 `
  --epochs 20 `
  --output-dir outputs/har_pamap2
```

## 2. WISDM2 / UCI-HAR Generalization

Use these as smaller or simpler IMU benchmarks after PAMAP2.

```powershell
python scripts/train_har_baseline.py `
  --source hf `
  --hf-repo monster-monash/WISDM2 `
  --epochs 20 `
  --output-dir outputs/har_wisdm2
```

```powershell
python scripts/train_har_baseline.py `
  --source hf `
  --hf-repo Beothuk/uci-har-federated `
  --epochs 20 `
  --output-dir outputs/har_uci
```

## 3. ShoeAlls Fine-Tuning

Export local ShoeAlls windows to an NPZ file with either:

- `windows`: shape `(samples, time, channels)` or `(samples, channels, time)`
- `labels`: integer or string class labels

Then fine-tune from the best public-data checkpoint:

```powershell
python scripts/train_har_baseline.py `
  --source npz `
  --local-npz data/shoealls_har_windows.npz `
  --checkpoint outputs/har_pamap2/best_har_model.pt `
  --freeze-encoder `
  --epochs 10 `
  --output-dir outputs/har_shoealls_finetune
```

Drop `--freeze-encoder` after the classifier head is stable.

## 4. Lightweight Transformer Classifier

The baseline model is `src.models.har_transformer.PatchTSTClassifier`.
It accepts normalized sensor windows as `(batch, channels, time)` and uses:

- patch projection
- CLS token pooling
- small Transformer encoder
- classification head

The model is intentionally separate from the existing multimodal gait network, so
PAMAP2/WISDM/UCI experiments cannot break the production multimodal API.

## 5. Forecasting / Anomaly Experiments

Chronos and TimesFM should stay outside the classification baseline. Use them later
for:

- next-window sensor forecasting
- reconstruction error / anomaly score
- gait drift monitoring over sessions

Do not make them the primary classifier until the HAR baseline and ShoeAlls
fine-tune results are recorded.
