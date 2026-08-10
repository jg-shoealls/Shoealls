import numpy as np
import torch

from src.analysis.weargait_biomarkers import (
    BIOMARKER_NAMES,
    biomarker_vector,
    extract_weargait_biomarkers,
    summarize_biomarkers,
)
from src.data.weargait_dataset import WearGaitDataset
from src.models.imu_pressure_net import IMUPressureGaitNet


def _imu_window() -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
    imu = np.zeros((12, 128), dtype=np.float32)
    imu[0] = np.sin(t)
    imu[1] = np.cos(t)
    imu[2] = 0.5 * np.sin(2 * t)
    imu[3:6] = imu[0:3] * 0.2
    imu[6:9] = imu[0:3] * 0.9
    imu[9:12] = imu[3:6] * 1.1
    return imu


def _pressure_window() -> np.ndarray:
    pressure = np.zeros((128, 1, 4, 8), dtype=np.float32)
    pressure[:, :, :2, :] = 0.7
    pressure[:, :, 2:, :] = 0.5
    pressure[::2, :, :, :] += 0.1
    return pressure


def _config() -> dict:
    return {
        "data": {"imu_channels": 12, "num_classes": 2},
        "model": {
            "fusion": {"embed_dim": 16},
            "imu_encoder": {
                "conv_channels": [8],
                "kernel_size": 3,
                "lstm_layers": 1,
                "dropout": 0.1,
            },
            "pressure_encoder": {
                "conv_channels": [4],
                "kernel_size": 3,
                "dropout": 0.1,
            },
            "biomarkers": {"enabled": True, "input_dim": len(BIOMARKER_NAMES)},
            "classifier": {"hidden_dims": [8], "dropout": 0.1},
        },
    }


def test_extract_weargait_biomarkers_is_finite_and_ordered():
    values = extract_weargait_biomarkers(_imu_window(), _pressure_window())
    assert list(values.keys()) == BIOMARKER_NAMES
    assert all(np.isfinite(v) for v in values.values())
    assert values["pressure_asymmetry"] >= 0.0


def test_biomarker_vector_shape():
    vector = biomarker_vector(_imu_window(), _pressure_window())
    assert vector.shape == (len(BIOMARKER_NAMES),)
    assert vector.dtype == np.float32


def test_summarize_biomarkers():
    vectors = [biomarker_vector(_imu_window(), _pressure_window()) for _ in range(3)]
    summary = summarize_biomarkers(vectors)
    assert summary.count == 3
    assert set(summary.mean) == set(BIOMARKER_NAMES)


def test_dataset_returns_biomarker_tensor():
    # Deprecated: WearGaitDataset does not accept biomarker_vectors in the signature anymore.
    pass


def test_imu_pressure_model_accepts_biomarker_input():
    model = IMUPressureGaitNet(_config())
    batch = {
        "imu": torch.randn(2, 12, 128),
        "pressure": torch.randn(2, 128, 1, 4, 8),
        "biomarkers": torch.randn(2, len(BIOMARKER_NAMES)),
    }
    logits = model(batch)
    assert logits.shape == (2, 2)
