"""Daphnet Freezing of Gait dataset loader for real-data training."""

import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ["normal_walking", "freezing_of_gait"]


class DaphnetDataset(Dataset):
    """Windowed segments from Daphnet accelerometer recordings.

    Maps 9 accel channels → 6-channel IMU (ankle xyz + lower_back xyz).
    Pressure and skeleton are zeroed (unavailable modality).
    Labels: 1 (normal walking) → 0, 2 (freeze) → 1. Label 0 excluded.
    """

    def __init__(
        self,
        data_dir: str,
        window: int = 128,
        stride: int = 64,
        grid_size: tuple = (16, 8),
        num_joints: int = 17,
    ):
        self.window = window
        self.stride = stride
        self.grid_size = grid_size
        self.num_joints = num_joints

        self.segments = []  # (T, 6) float32
        self.labels = []  # int

        for path in sorted(pathlib.Path(data_dir).glob("*.txt")):
            raw = np.loadtxt(path)  # (N, 11)
            accel = raw[:, 1:10].astype(np.float32)  # 9 channels
            labels = raw[:, 10].astype(int)

            # Normalise to roughly [-1, 1]
            accel = accel / 1000.0

            # Keep ankle xyz (cols 0-2) + lower_back xyz (cols 6-8) → 6 ch
            imu6 = np.concatenate([accel[:, 0:3], accel[:, 6:9]], axis=1)

            self._window_recording(imu6, labels)

    def _window_recording(self, imu: np.ndarray, labels: np.ndarray):
        n = len(imu)
        for start in range(0, n - self.window + 1, self.stride):
            seg_labels = labels[start : start + self.window]
            # Skip windows that contain transition frames (label 0)
            if np.any(seg_labels == 0):
                continue
            # Majority vote for window label
            majority = int(np.bincount(seg_labels).argmax())
            self.segments.append(imu[start : start + self.window].copy())
            self.labels.append(majority - 1)  # 1→0, 2→1

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        imu = torch.from_numpy(self.segments[idx]).T  # (6, T)
        h, w = self.grid_size
        pressure = torch.zeros(self.window, 1, h, w)  # (T, 1, H, W)
        skeleton = torch.zeros(3, self.window, self.num_joints)  # (3, T, J)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"imu": imu, "pressure": pressure, "skeleton": skeleton, "label": label}
