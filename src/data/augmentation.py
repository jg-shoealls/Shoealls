"""Data augmentation transforms for multimodal gait data.

보행 데이터 증강 변환 모듈.
시간 왜곡, 노이즈 주입, 센서 드롭아웃, 좌우 반전, 크기 스케일링,
랜덤 시간 자르기 등 다양한 증강 기법을 제공합니다.

모든 변환은 numpy 배열 기반으로 동작하며, ``Compose`` 클래스를 통해
파이프라인으로 연결할 수 있습니다.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


class TimeWarp:
    """시간 축 왜곡 (신축/압축).

    Stretch or compress the temporal axis by resampling with a smooth
    random warping curve.

    Args:
        sigma: Controls the magnitude of warping. Higher values produce
            more aggressive distortion.
        num_knots: Number of control knots for the warping spline.
    """

    def __init__(self, sigma: float = 0.2, num_knots: int = 4):
        self.sigma = sigma
        self.num_knots = num_knots

    def __call__(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Apply time warping.

        Args:
            x: Input array with time along axis 0, shape ``(T, ...)``.
            rng: Optional numpy random generator.

        Returns:
            Time-warped array with the same shape as input.
        """
        rng = rng or np.random.default_rng()
        T = x.shape[0]
        if T < 2:
            return x

        # Build a smooth warping path
        orig_steps = np.arange(T)
        knot_positions = np.linspace(0, T - 1, self.num_knots + 2)
        knot_offsets = np.concatenate([
            [0],
            rng.normal(0, self.sigma, self.num_knots) * T,
            [0],
        ])
        warped_knots = knot_positions + knot_offsets
        warped_knots = np.clip(warped_knots, 0, T - 1)
        warped_knots = np.sort(warped_knots)

        warp_fn = interp1d(knot_positions, warped_knots, kind="cubic", fill_value="extrapolate")
        warped_steps = warp_fn(orig_steps)
        warped_steps = np.clip(warped_steps, 0, T - 1)

        # Resample each non-time axis
        flat = x.reshape(T, -1)
        resampled = np.zeros_like(flat)
        for col in range(flat.shape[1]):
            f = interp1d(orig_steps, flat[:, col], kind="linear", fill_value="extrapolate")
            resampled[:, col] = f(warped_steps)

        return resampled.reshape(x.shape)


class GaussianNoise:
    """가우시안 노이즈 주입.

    Add independent Gaussian noise to every element.

    Args:
        std: Standard deviation of the noise relative to the input's
            standard deviation when ``relative=True``, or as absolute
            value otherwise.
        relative: If True, noise std is ``std * x.std()``.
    """

    def __init__(self, std: float = 0.05, relative: bool = True):
        self.std = std
        self.relative = relative

    def __call__(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        sigma = self.std * x.std() if self.relative else self.std
        return x + rng.normal(0, sigma, x.shape).astype(x.dtype)


class RandomSensorDropout:
    """랜덤 센서(모달리티 채널) 드롭아웃.

    Zero out randomly selected channels along a specified axis to simulate
    sensor failures and encourage modality-independent representations.

    Args:
        p: Probability of each channel being zeroed out.
        channel_axis: Axis index representing channels.
    """

    def __init__(self, p: float = 0.2, channel_axis: int = -1):
        self.p = p
        self.channel_axis = channel_axis

    def __call__(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        axis = self.channel_axis % x.ndim
        n_channels = x.shape[axis]
        mask_shape = [1] * x.ndim
        mask_shape[axis] = n_channels
        mask = (rng.random(mask_shape) > self.p).astype(x.dtype)
        return x * mask


class LeftRightFlip:
    """좌우 반전 (보행 대칭 미러링).

    Mirror the gait pattern by swapping left and right sides.
    Applicable to skeleton data where joint indices have a known
    left/right pairing, and to pressure maps along the lateral axis.

    Args:
        flip_pairs: List of ``(left_idx, right_idx)`` joint pairs for
            skeleton data. If None, a default COCO-style 17-joint mapping
            is used.
        data_type: One of ``'skeleton'``, ``'pressure'``, or ``'imu'``.
    """

    # Default COCO-17 skeleton left-right pairs (0-indexed)
    DEFAULT_FLIP_PAIRS = [
        (3, 6), (4, 7), (5, 8),    # shoulder, elbow, wrist
        (9, 12), (10, 13), (11, 14),  # hip, knee, ankle
        (15, 16),                    # feet
    ]

    def __init__(
        self,
        flip_pairs: list[tuple[int, int]] | None = None,
        data_type: str = "skeleton",
    ):
        self.flip_pairs = flip_pairs or self.DEFAULT_FLIP_PAIRS
        self.data_type = data_type

    def __call__(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        x = x.copy()

        if self.data_type == "skeleton":
            # Expected shape: (T, J, 3) or (C, T, J)
            return self._flip_skeleton(x)
        elif self.data_type == "pressure":
            # Expected shape: (T, H, W) or (T, 1, H, W)
            return self._flip_pressure(x)
        elif self.data_type == "imu":
            # Expected shape: (T, 6) -- negate lateral axes
            return self._flip_imu(x)
        else:
            raise ValueError(f"Unknown data_type: {self.data_type}")

    def _flip_skeleton(self, x: np.ndarray) -> np.ndarray:
        """Swap left/right joints and negate the lateral (x) coordinate."""
        if x.ndim == 3 and x.shape[2] == 3:
            # (T, J, 3) format
            for l_idx, r_idx in self.flip_pairs:
                if l_idx < x.shape[1] and r_idx < x.shape[1]:
                    x[:, l_idx, :], x[:, r_idx, :] = (
                        x[:, r_idx, :].copy(),
                        x[:, l_idx, :].copy(),
                    )
            x[:, :, 0] *= -1  # negate x-axis
        elif x.ndim == 3 and x.shape[0] == 3:
            # (C=3, T, J) preprocessed format
            for l_idx, r_idx in self.flip_pairs:
                if l_idx < x.shape[2] and r_idx < x.shape[2]:
                    x[:, :, l_idx], x[:, :, r_idx] = (
                        x[:, :, r_idx].copy(),
                        x[:, :, l_idx].copy(),
                    )
            x[0, :, :] *= -1  # negate x-axis (channel 0)
        return x

    @staticmethod
    def _flip_pressure(x: np.ndarray) -> np.ndarray:
        """Flip pressure map along the lateral (width) axis."""
        if x.ndim == 3:
            return np.flip(x, axis=2).copy()
        elif x.ndim == 4:
            return np.flip(x, axis=3).copy()
        return x

    @staticmethod
    def _flip_imu(x: np.ndarray) -> np.ndarray:
        """Negate lateral acceleration and gyroscope axes."""
        # Channels: [ax, ay, az, gx, gy, gz]
        # Lateral axes: ax (idx 0), gx (idx 3)
        if x.ndim == 2 and x.shape[-1] == 6:
            x[:, 0] *= -1  # ax
            x[:, 3] *= -1  # gx
        elif x.ndim == 2 and x.shape[0] == 6:
            # Preprocessed (C, T) format
            x[0, :] *= -1
            x[3, :] *= -1
        return x


class MagnitudeScale:
    """크기 스케일링.

    Multiply all values by a random scalar drawn from a uniform
    distribution around 1.0.

    Args:
        low: Lower bound of the scale factor.
        high: Upper bound of the scale factor.
    """

    def __init__(self, low: float = 0.8, high: float = 1.2):
        self.low = low
        self.high = high

    def __call__(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        scale = rng.uniform(self.low, self.high)
        return (x * scale).astype(x.dtype)


class RandomTimeCrop:
    """랜덤 시간 자르기 및 패딩.

    Crop a random contiguous window of ``crop_length`` frames from the
    temporal axis (axis 0). If the input is shorter than ``crop_length``,
    it is zero-padded at the end.

    Args:
        crop_length: Target number of frames.
        pad_value: Value used for padding when input is shorter.
    """

    def __init__(self, crop_length: int = 128, pad_value: float = 0.0):
        self.crop_length = crop_length
        self.pad_value = pad_value

    def __call__(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        T = x.shape[0]

        if T > self.crop_length:
            start = rng.integers(0, T - self.crop_length)
            return x[start : start + self.crop_length].copy()
        elif T < self.crop_length:
            pad_width = [(0, self.crop_length - T)] + [(0, 0)] * (x.ndim - 1)
            return np.pad(x, pad_width, mode="constant", constant_values=self.pad_value)
        else:
            return x.copy()


class Compose:
    """증강 파이프라인 구성 클래스.

    Chain multiple augmentation transforms into a sequential pipeline.
    Each transform is applied with an independent application probability.

    Args:
        transforms: List of ``(transform, probability)`` tuples.
            Each transform is applied only if a uniform draw is below
            the given probability.
        seed: Optional seed for reproducibility.

    Example::

        augmentor = Compose([
            (TimeWarp(sigma=0.2), 0.5),
            (GaussianNoise(std=0.03), 0.8),
            (MagnitudeScale(0.9, 1.1), 0.5),
        ], seed=42)

        augmented = augmentor(data)
    """

    def __init__(
        self,
        transforms: list[tuple[object, float]],
        seed: int | None = None,
    ):
        self.transforms = transforms
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the augmentation pipeline.

        Args:
            x: Input numpy array.

        Returns:
            Augmented numpy array.
        """
        for transform, prob in self.transforms:
            if self.rng.random() < prob:
                x = transform(x, rng=self.rng)
        return x

    def __repr__(self) -> str:
        lines = [f"Compose(["]
        for t, p in self.transforms:
            lines.append(f"  ({t.__class__.__name__}(...), p={p}),")
        lines.append("])")
        return "\n".join(lines)
