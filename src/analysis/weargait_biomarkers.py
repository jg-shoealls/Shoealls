"""Explicit WearGait-PD gait biomarkers for model input and reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BIOMARKER_NAMES = [
    "acc_rms",
    "acc_variability",
    "gyro_rms",
    "gyro_variability",
    "right_left_acc_asymmetry",
    "right_left_gyro_asymmetry",
    "pressure_asymmetry",
    "double_support_ratio",
    "heel_pressure_ratio",
    "forefoot_pressure_ratio",
    "midfoot_pressure_ratio",
    "cop_path_length",
]


BIOMARKER_INFO = {
    "acc_rms": (
        "IMU",
        "ankle acceleration magnitude; lower values can reflect bradykinetic gait",
    ),
    "acc_variability": ("IMU", "stride-to-stride acceleration variability"),
    "gyro_rms": ("IMU", "ankle angular velocity magnitude"),
    "gyro_variability": ("IMU", "angular velocity rhythm variability"),
    "right_left_acc_asymmetry": ("IMU", "left-right ankle acceleration imbalance"),
    "right_left_gyro_asymmetry": ("IMU", "left-right ankle angular velocity imbalance"),
    "pressure_asymmetry": ("PRESSURE", "left-right plantar loading imbalance"),
    "double_support_ratio": ("PRESSURE", "fraction of frames with both feet loaded"),
    "heel_pressure_ratio": ("PRESSURE", "heel loading fraction"),
    "forefoot_pressure_ratio": ("PRESSURE", "forefoot loading fraction"),
    "midfoot_pressure_ratio": ("PRESSURE", "midfoot loading fraction"),
    "cop_path_length": ("PRESSURE", "normalized center-of-pressure path length"),
}


@dataclass(frozen=True)
class WearGaitBiomarkerSummary:
    """Summary statistics for a group of WearGait biomarker vectors."""

    count: int
    mean: dict[str, float]
    std: dict[str, float]


def extract_weargait_biomarkers(
    imu: np.ndarray, pressure: np.ndarray
) -> dict[str, float]:
    """Extract explicit biomarkers from one WearGait window.

    Args:
        imu: Array shaped ``(12, T)`` with right ankle accel/gyro followed by
            left ankle accel/gyro.
        pressure: Array shaped ``(T, 1, 4, 8)`` or compatible flattened pressure
            frames. The first 16 sensors are left foot, the next 16 are right foot.

    Returns:
        Dict in ``BIOMARKER_NAMES`` order containing finite float values.
    """

    imu = np.asarray(imu, dtype=np.float32)
    pressure = _pressure_to_feet(np.asarray(pressure, dtype=np.float32))

    r_acc = _channels(imu, 0, 3)
    r_gyr = _channels(imu, 3, 6)
    l_acc = _channels(imu, 6, 9)
    l_gyr = _channels(imu, 9, 12)

    acc_mag = _magnitude(np.concatenate([r_acc, l_acc], axis=0))
    gyro_mag = _magnitude(np.concatenate([r_gyr, l_gyr], axis=0))

    left_pressure = pressure[:, 0]
    right_pressure = pressure[:, 1]
    total_pressure = left_pressure + right_pressure

    left_force = left_pressure.sum(axis=(1, 2))
    right_force = right_pressure.sum(axis=(1, 2))
    total_force = left_force + right_force

    heel = total_pressure[:, 3, :].sum(axis=1)
    forefoot = total_pressure[:, :2, :].sum(axis=(1, 2))
    midfoot = total_pressure[:, 2, :].sum(axis=1)

    return _finite_dict(
        {
            "acc_rms": _rms(acc_mag),
            "acc_variability": _cv(acc_mag),
            "gyro_rms": _rms(gyro_mag),
            "gyro_variability": _cv(gyro_mag),
            "right_left_acc_asymmetry": _asymmetry(
                _rms(_magnitude(r_acc)), _rms(_magnitude(l_acc))
            ),
            "right_left_gyro_asymmetry": _asymmetry(
                _rms(_magnitude(r_gyr)), _rms(_magnitude(l_gyr))
            ),
            "pressure_asymmetry": _asymmetry(
                float(left_force.sum()), float(right_force.sum())
            ),
            "double_support_ratio": _double_support_ratio(left_force, right_force),
            "heel_pressure_ratio": _safe_ratio(
                float(heel.sum()), float(total_force.sum())
            ),
            "forefoot_pressure_ratio": _safe_ratio(
                float(forefoot.sum()), float(total_force.sum())
            ),
            "midfoot_pressure_ratio": _safe_ratio(
                float(midfoot.sum()), float(total_force.sum())
            ),
            "cop_path_length": _cop_path_length(total_pressure),
        }
    )


def biomarker_vector(imu: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Return biomarkers as a stable float32 vector."""

    values = extract_weargait_biomarkers(imu, pressure)
    return np.asarray([values[name] for name in BIOMARKER_NAMES], dtype=np.float32)


def summarize_biomarkers(vectors: list[np.ndarray]) -> WearGaitBiomarkerSummary:
    """Compute mean/std by biomarker name."""

    if not vectors:
        zeros = {name: 0.0 for name in BIOMARKER_NAMES}
        return WearGaitBiomarkerSummary(count=0, mean=zeros, std=zeros)

    arr = np.asarray(vectors, dtype=np.float32)
    mean = {name: float(arr[:, idx].mean()) for idx, name in enumerate(BIOMARKER_NAMES)}
    std = {name: float(arr[:, idx].std()) for idx, name in enumerate(BIOMARKER_NAMES)}
    return WearGaitBiomarkerSummary(count=int(arr.shape[0]), mean=mean, std=std)


def _channels(imu: np.ndarray, start: int, end: int) -> np.ndarray:
    if imu.ndim != 2 or imu.shape[0] < end:
        return np.zeros(
            (end - start, max(1, imu.shape[-1] if imu.ndim else 1)), dtype=np.float32
        )
    return imu[start:end]


def _pressure_to_feet(pressure: np.ndarray) -> np.ndarray:
    if pressure.ndim == 4:
        flat = pressure[:, 0].reshape(pressure.shape[0], -1)
    elif pressure.ndim == 3:
        flat = pressure.reshape(pressure.shape[0], -1)
    elif pressure.ndim == 2:
        flat = pressure
    else:
        flat = pressure.reshape(max(1, pressure.shape[0]), -1)

    if flat.shape[1] < 32:
        padded = np.zeros((flat.shape[0], 32), dtype=np.float32)
        padded[:, : flat.shape[1]] = flat
        flat = padded

    left = flat[:, :16].reshape(flat.shape[0], 4, 4)
    right = flat[:, 16:32].reshape(flat.shape[0], 4, 4)
    return np.stack([left, right], axis=1)


def _magnitude(channels: np.ndarray) -> np.ndarray:
    if channels.size == 0:
        return np.zeros(1, dtype=np.float32)
    return np.sqrt(np.sum(channels.astype(np.float32) ** 2, axis=0))


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values**2)))


def _cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size < 2:
        return 0.0
    mean = float(np.mean(np.abs(values)))
    if mean <= 1e-8:
        return 0.0
    return float(np.std(values) / mean)


def _asymmetry(left: float, right: float) -> float:
    return _safe_ratio(abs(left - right), abs(left) + abs(right))


def _safe_ratio(num: float, den: float) -> float:
    if abs(den) <= 1e-8:
        return 0.0
    return float(num / den)


def _double_support_ratio(left_force: np.ndarray, right_force: np.ndarray) -> float:
    threshold = 0.05 * max(float(np.max(left_force)), float(np.max(right_force)), 1e-8)
    both = (left_force > threshold) & (right_force > threshold)
    return float(np.mean(both)) if both.size else 0.0


def _cop_path_length(total_pressure: np.ndarray) -> float:
    if total_pressure.size == 0:
        return 0.0

    h, w = total_pressure.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    force = total_pressure.sum(axis=(1, 2))
    valid = force > 1e-8
    if valid.sum() < 2:
        return 0.0

    x = (total_pressure * xx).sum(axis=(1, 2)) / np.maximum(force, 1e-8)
    y = (total_pressure * yy).sum(axis=(1, 2)) / np.maximum(force, 1e-8)
    dx = np.diff(x[valid])
    dy = np.diff(y[valid])
    scale = max(float(h + w), 1.0)
    return float(np.sqrt(dx**2 + dy**2).sum() / scale)


def _finite_dict(values: dict[str, float]) -> dict[str, float]:
    out = {}
    for name in BIOMARKER_NAMES:
        value = float(values.get(name, 0.0))
        out[name] = value if np.isfinite(value) else 0.0
    return out
