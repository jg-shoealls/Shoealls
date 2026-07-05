"""Dataset utilities for WearGait-PD CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IMU_COLS = [
    "R_Ankle_Acc_X", "R_Ankle_Acc_Y", "R_Ankle_Acc_Z",
    "R_Ankle_Gyr_X", "R_Ankle_Gyr_Y", "R_Ankle_Gyr_Z",
    "L_Ankle_Acc_X", "L_Ankle_Acc_Y", "L_Ankle_Acc_Z",
    "L_Ankle_Gyr_X", "L_Ankle_Gyr_Y", "L_Ankle_Gyr_Z",
]

PRESSURE_COLS = (
    [f"LPressure{i}" for i in range(1, 17)]
    + [f"RPressure{i}" for i in range(1, 17)]
)


class WearGaitDataset(Dataset):
    """Windowed WearGait-PD IMU and plantar pressure samples."""

    def __init__(self, imu_windows, pressure_windows, labels, biomarkers=None):
        self.imu_windows = imu_windows
        self.pressure_windows = pressure_windows
        self.labels = labels
        self.biomarkers = biomarkers

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out = {
            "imu": torch.as_tensor(self.imu_windows[idx], dtype=torch.float32),
            "pressure": torch.as_tensor(self.pressure_windows[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

        if self.biomarkers is not None:
            if isinstance(self.biomarkers[idx], dict):
                bm_t = torch.tensor(list(self.biomarkers[idx].values()), dtype=torch.float32)
            else:
                bm_t = torch.tensor(self.biomarkers[idx], dtype=torch.float32)
            out["biomarkers"] = bm_t

        return out


def load_weargait_by_subject(data_dir: str | Path, window: int = 128, stride: int = 64) -> dict:
    """Load WearGait-PD self-paced CSVs grouped by subject id.

    Returns:
        Dict mapping subject id to ``(imu_windows, pressure_windows, labels)``.
    """
    data_path = Path(data_dir)
    subjects: dict[str, tuple[list[np.ndarray], list[np.ndarray], list[int]]] = {}

    for path in sorted(data_path.rglob("*.csv")):
        name = path.name.lower()
        if "selfpace" not in name or "manifest" in name:
            continue

        label = _label_from_name(path.name)
        if label is None:
            continue

        try:
            imu, pressure = _load_csv_windows(path, window=window, stride=stride)
        except ValueError as exc:
            print(f"[skip] {path}: {exc}")
            continue

        subject_id = _subject_id(path)
        if subject_id not in subjects:
            subjects[subject_id] = ([], [], [])
        subjects[subject_id][0].extend(imu)
        subjects[subject_id][1].extend(pressure)
        subjects[subject_id][2].extend([label] * len(imu))

    return subjects


def _load_csv_windows(path: Path, window: int, stride: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    header = pd.read_csv(path, nrows=0).columns
    missing = [col for col in [*IMU_COLS, *PRESSURE_COLS] if col not in header]
    if missing:
        raise ValueError(f"missing required columns: {missing[:5]}")

    df = pd.read_csv(path, usecols=[*IMU_COLS, *PRESSURE_COLS], low_memory=False)
    values = df.apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").fillna(0.0)

    imu = values[IMU_COLS].to_numpy(dtype=np.float32)
    pressure = values[PRESSURE_COLS].to_numpy(dtype=np.float32)

    imu = _standardize(imu)
    pressure = pressure / max(float(np.nanmax(np.abs(pressure))), 1.0)

    imu_windows: list[np.ndarray] = []
    pressure_windows: list[np.ndarray] = []
    for start in range(0, len(values) - window + 1, stride):
        end = start + window
        imu_win = imu[start:end].T
        pressure_win = pressure[start:end].reshape(window, 1, 4, 8)
        imu_windows.append(imu_win.astype(np.float32, copy=False))
        pressure_windows.append(pressure_win.astype(np.float32, copy=False))

    if not imu_windows:
        raise ValueError(f"not enough rows for window={window}")
    return imu_windows, pressure_windows


def _standardize(array: np.ndarray) -> np.ndarray:
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    return (array - mean) / np.maximum(std, 1e-6)


def _label_from_name(name: str) -> int | None:
    lower = name.lower()
    if lower.startswith(("whc", "hc", "control")):
        return 0
    if lower.startswith(("wpd", "pd", "nls")):
        return 1
    return None


def _subject_id(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_selfpace", "-selfpace"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem
