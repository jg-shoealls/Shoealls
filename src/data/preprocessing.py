"""Preprocessing utilities for multimodal gait data."""

import numpy as np
from scipy.signal import resample


def preprocess_imu(data: np.ndarray, target_length: int = 128) -> np.ndarray:
    """Preprocess IMU sensor data (accelerometer + gyroscope).

    Args:
        data: Raw IMU data of shape (T, 6) - [ax, ay, az, gx, gy, gz]
        target_length: Target sequence length after resampling.

    Returns:
        Preprocessed IMU data of shape (6, target_length).
    """
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"Expected IMU data shape (T, 6), got {data.shape}")

    # Resample to target length
    resampled = resample(data, target_length, axis=0)

    # Normalize per channel (z-score)
    mean = resampled.mean(axis=0, keepdims=True)
    std = resampled.std(axis=0, keepdims=True) + 1e-8
    normalized = (resampled - mean) / std

    # Transpose to (channels, time)
    return normalized.T.astype(np.float32)


def preprocess_pressure(
    data: np.ndarray,
    target_length: int = 128,
    grid_size: tuple = (16, 8),
) -> np.ndarray:
    """Preprocess plantar pressure sensor data.

    Args:
        data: Raw pressure data of shape (T, H, W) or (T, H*W).
        target_length: Target sequence length.
        grid_size: Pressure sensor grid dimensions (H, W).

    Returns:
        Preprocessed pressure data of shape (target_length, 1, H, W).
    """
    h, w = grid_size

    if data.ndim == 2:
        data = data.reshape(-1, h, w)

    if data.ndim != 3:
        raise ValueError(f"Expected pressure data with 2 or 3 dims, got {data.ndim}")

    # Resample temporal dimension
    resampled = resample(data, target_length, axis=0)

    # Min-max normalization per frame
    frame_min = resampled.min(axis=(1, 2), keepdims=True)
    frame_max = resampled.max(axis=(1, 2), keepdims=True)
    normalized = (resampled - frame_min) / (frame_max - frame_min + 1e-8)

    # Add channel dimension: (T, 1, H, W)
    return normalized[:, np.newaxis, :, :].astype(np.float32)


def preprocess_skeleton(
    data: np.ndarray,
    target_length: int = 128,
    num_joints: int = 17,
) -> np.ndarray:
    """Preprocess skeleton joint position data.

    Args:
        data: Raw skeleton data of shape (T, J, 3) - [x, y, z] per joint.
        target_length: Target sequence length.
        num_joints: Expected number of joints.

    Returns:
        Preprocessed skeleton data of shape (3, target_length, num_joints).
    """
    if data.ndim != 3 or data.shape[1] != num_joints or data.shape[2] != 3:
        raise ValueError(
            f"Expected skeleton data shape (T, {num_joints}, 3), got {data.shape}"
        )

    # Resample temporal dimension
    resampled = resample(data, target_length, axis=0)

    # Center skeleton at hip joint (joint 0) per frame
    hip = resampled[:, 0:1, :]
    centered = resampled - hip

    # Normalize by body scale (max distance from hip)
    scale = np.max(np.linalg.norm(centered, axis=2), axis=1, keepdims=True)
    scale = np.expand_dims(scale, axis=2) + 1e-8
    normalized = centered / scale

    # Reshape to (C=3, T, J)
    return normalized.transpose(2, 0, 1).astype(np.float32)
