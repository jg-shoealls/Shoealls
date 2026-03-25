"""PyTorch dataset for multimodal gait data."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_imu, preprocess_pressure, preprocess_skeleton


class MultimodalGaitDataset(Dataset):
    """Dataset combining IMU, pressure, and skeleton modalities."""

    def __init__(
        self,
        data_dict: dict,
        sequence_length: int = 128,
        grid_size: tuple = (16, 8),
        num_joints: int = 17,
    ):
        """
        Args:
            data_dict: Dictionary with 'imu', 'pressure', 'skeleton', 'labels'.
            sequence_length: Target sequence length for all modalities.
            grid_size: Pressure sensor grid (H, W).
            num_joints: Number of skeleton joints.
        """
        self.imu_data = data_dict["imu"]
        self.pressure_data = data_dict["pressure"]
        self.skeleton_data = data_dict["skeleton"]
        self.labels = data_dict["labels"]
        self.sequence_length = sequence_length
        self.grid_size = grid_size
        self.num_joints = num_joints

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        imu = preprocess_imu(self.imu_data[idx], self.sequence_length)
        pressure = preprocess_pressure(
            self.pressure_data[idx], self.sequence_length, self.grid_size
        )
        skeleton = preprocess_skeleton(
            self.skeleton_data[idx], self.sequence_length, self.num_joints
        )
        label = self.labels[idx]

        return {
            "imu": torch.from_numpy(imu),
            "pressure": torch.from_numpy(pressure),
            "skeleton": torch.from_numpy(skeleton),
            "label": torch.tensor(label, dtype=torch.long),
        }
