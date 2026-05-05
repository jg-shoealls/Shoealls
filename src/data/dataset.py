"""PyTorch dataset — 신발 전용 멀티모달 보행 데이터.

신발에서만 추출 가능한 3 모달리티:
    imu      : (6, T)         발목 IMU — 가속도 3축 + 자이로 3축
    pressure : (T, 1, H, W)   인솔 족저압 그리드
    mag_baro : (5, T)         지자기 4채널 + 기압 1채널
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    preprocess_imu,
    preprocess_pressure,
    preprocess_magnetometer,
    preprocess_barometer,
)


class GaitAugmentation:
    """학습 전용 온라인 증강."""

    def __init__(
        self,
        noise_std: float = 0.04,
        amplitude_range: tuple = (0.85, 1.15),
        time_shift_max: int = 8,
        frame_dropout: float = 0.05,
    ):
        self.noise_std = noise_std
        self.amplitude_range = amplitude_range
        self.time_shift_max = time_shift_max
        self.frame_dropout = frame_dropout

    def augment_imu(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        scale = rng.uniform(*self.amplitude_range, size=(x.shape[0], 1)).astype(np.float32)
        x = x * scale + (self.noise_std * rng.standard_normal(x.shape)).astype(np.float32)
        shift = int(rng.integers(-self.time_shift_max, self.time_shift_max + 1))
        return np.roll(x, shift, axis=1) if shift else x

    def augment_pressure(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = x + (self.noise_std * 0.5 * rng.standard_normal(x.shape)).astype(np.float32)
        if self.frame_dropout > 0:
            mask = (rng.random(x.shape[0]) > self.frame_dropout).astype(np.float32)
            x = x * mask[:, None, None, None]
        return np.clip(x, 0.0, 1.0)

    def augment_mag_baro(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return x + (self.noise_std * 0.5 * rng.standard_normal(x.shape)).astype(np.float32)


class MultimodalGaitDataset(Dataset):
    """신발 전용 3-모달리티 보행 데이터셋.

    Args:
        data_dict : generate_prodromal_dataset() 반환값
        augment   : True이면 학습용 증강 적용
        aug_cfg   : GaitAugmentation 파라미터 오버라이드
    """

    def __init__(
        self,
        data_dict: dict,
        sequence_length: int = 128,
        grid_size: tuple = (16, 8),
        num_joints: int = 17,   # 하위 호환성 — 내부에서 사용하지 않음
        augment: bool = False,
        aug_cfg: dict = None,
    ):
        self.imu_data  = data_dict["imu"]
        self.pres_data = data_dict["pressure"]
        self.mag_data  = data_dict["magnetometer"]
        self.baro_data = data_dict["barometer"]
        self.labels    = data_dict["labels"]

        self.seq_len   = sequence_length
        self.grid_size = grid_size
        self.augment   = augment
        self._aug      = GaitAugmentation(**(aug_cfg or {})) if augment else None
        self._rng      = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        imu  = preprocess_imu(self.imu_data[idx], self.seq_len)          # (6, T)
        pres = preprocess_pressure(self.pres_data[idx], self.seq_len, self.grid_size)  # (T,1,H,W)
        mag  = preprocess_magnetometer(self.mag_data[idx], self.seq_len)  # (4, T)
        baro = preprocess_barometer(self.baro_data[idx], self.seq_len)    # (1, T)
        mag_baro = np.concatenate([mag, baro], axis=0)                    # (5, T)

        if self.augment and self._aug is not None:
            imu      = self._aug.augment_imu(imu, self._rng)
            pres     = self._aug.augment_pressure(pres, self._rng)
            mag_baro = self._aug.augment_mag_baro(mag_baro, self._rng)

        return {
            "imu":      torch.from_numpy(imu),
            "pressure": torch.from_numpy(pres),
            "mag_baro": torch.from_numpy(mag_baro),
            "label":    torch.tensor(self.labels[idx], dtype=torch.long),
        }
