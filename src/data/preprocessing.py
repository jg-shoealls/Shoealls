"""Preprocessing utilities for multimodal gait data."""

import numpy as np


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _sanitize(data: np.ndarray) -> np.ndarray:
    """NaN → 선형 보간, Inf → 유한 최댓값으로 클리핑."""
    data = data.copy().astype(np.float32)

    # ± Inf: 유한 값 범위의 ±3σ 또는 ±1e4 내로 클리핑
    if not np.isfinite(data).all():
        finite_vals = data[np.isfinite(data)]
        if len(finite_vals) > 0:
            clip_val = float(np.abs(finite_vals).max()) * 2 + 1e-6
            clip_val = min(clip_val, 1e4)   # 센서 값이 1e4 초과하는 경우 없다고 가정
        else:
            clip_val = 1e4
        data = np.clip(np.nan_to_num(data, nan=np.nan, posinf=clip_val, neginf=-clip_val),
                       -clip_val, clip_val)

    # NaN → 선형 보간 (시간 축 = axis 0)
    if np.isnan(data).any():
        for ch in range(data.shape[-1] if data.ndim >= 2 else 1):
            if data.ndim == 1:
                col = data
            elif data.ndim == 2:
                col = data[:, ch]
            else:
                # (T, ...) 형태를 (T, -1)로 flatten해서 보간
                flat = data.reshape(len(data), -1)
                for c in range(flat.shape[1]):
                    col = flat[:, c]
                    nan_idx = np.where(np.isnan(col))[0]
                    if len(nan_idx) == 0:
                        continue
                    ok_idx = np.where(~np.isnan(col))[0]
                    if len(ok_idx) == 0:
                        flat[:, c] = 0.0
                    else:
                        flat[:, c] = np.interp(np.arange(len(col)), ok_idx, col[ok_idx])
                data = flat.reshape(data.shape)
                return data

            nan_idx = np.where(np.isnan(col))[0]
            if len(nan_idx) == 0:
                continue
            ok_idx = np.where(~np.isnan(col))[0]
            if len(ok_idx) == 0:
                col[:] = 0.0
            else:
                interp = np.interp(np.arange(len(col)), ok_idx, col[ok_idx])
                if data.ndim == 1:
                    data[:] = interp
                else:
                    data[:, ch] = interp

    return data


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_imu(data: np.ndarray, target_length: int = 128) -> np.ndarray:
    """Preprocess IMU sensor data (accelerometer + gyroscope).

    Args:
        data: Raw IMU data of shape (T, 6) or (T, 7) — [ts?, ax, ay, az, gx, gy, gz].
              If 7 columns, the first column is assumed to be a timestamp and is dropped.
        target_length: Target sequence length after resampling.

    Returns:
        Preprocessed IMU data of shape (6, target_length).
    """
    data = np.array(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D IMU array (T, 6), got shape {data.shape}")

    # 타임스탬프 열 자동 제거
    if data.shape[1] == 7:
        data = data[:, 1:]
    if data.shape[1] != 6:
        raise ValueError(
            f"Expected IMU data with 6 channels [ax,ay,az,gx,gy,gz], "
            f"got {data.shape[1]} columns. "
            f"Use imu_cols parameter in FolderDataAdapter to select the right columns."
        )

    # NaN / Inf 정리
    data = _sanitize(data)

    # 시간 축 리샘플
    from scipy.signal import resample
    resampled = resample(data, target_length, axis=0)

    # Z-score 정규화 (채널별)
    mean = resampled.mean(axis=0, keepdims=True)
    std  = resampled.std(axis=0, keepdims=True) + 1e-8
    normalized = (resampled - mean) / std

    return normalized.T.astype(np.float32)   # (6, T)


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
    data = np.array(data, dtype=np.float32)

    if data.ndim == 2 and data.shape[1] == h * w:
        data = data.reshape(-1, h, w)
    elif data.ndim == 2:
        raise ValueError(
            f"Pressure flat data has {data.shape[1]} columns, "
            f"expected {h}*{w}={h*w} for grid_size={grid_size}."
        )
    if data.ndim != 3:
        raise ValueError(f"Expected pressure array with 2 or 3 dims, got {data.ndim}")

    # NaN → 0 (발을 떼서 측정값 없음과 동일하게 처리)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # 시간 축 리샘플
    from scipy.signal import resample
    resampled = resample(data, target_length, axis=0)

    # Min-max 정규화 (프레임별)
    frame_min = resampled.min(axis=(1, 2), keepdims=True)
    frame_max = resampled.max(axis=(1, 2), keepdims=True)
    normalized = (resampled - frame_min) / (frame_max - frame_min + 1e-8)

    return normalized[:, np.newaxis, :, :].astype(np.float32)   # (T, 1, H, W)


def preprocess_skeleton(
    data: np.ndarray,
    target_length: int = 128,
    num_joints: int = 17,
) -> np.ndarray:
    """Preprocess skeleton joint position data.

    Args:
        data: Raw skeleton data of shape (T, J, 3) or (T, J, 2).
              2-D skeletons are zero-padded to 3-D.
        target_length: Target sequence length.
        num_joints: Expected number of joints.

    Returns:
        Preprocessed skeleton data of shape (3, target_length, num_joints).
    """
    data = np.array(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(f"Expected 3-D skeleton array (T, J, C), got shape {data.shape}")

    T, J, C = data.shape
    if J != num_joints:
        raise ValueError(
            f"Expected {num_joints} joints, got {J}. "
            f"Adjust num_joints parameter."
        )

    # 2D 스켈레톤 → z=0 패딩
    if C == 2:
        data = np.concatenate([data, np.zeros((T, J, 1), dtype=np.float32)], axis=2)
    elif C != 3:
        raise ValueError(f"Skeleton coords must be 2 or 3, got {C}.")

    # NaN → 선형 보간
    data = _sanitize(data)

    # 시간 축 리샘플
    from scipy.signal import resample
    resampled = resample(data, target_length, axis=0)

    # Hip(0번 관절) 중심 정규화
    hip   = resampled[:, 0:1, :]
    centered = resampled - hip

    # 신체 스케일 정규화
    scale = np.max(np.linalg.norm(centered, axis=2), axis=1, keepdims=True)
    scale = np.expand_dims(scale, axis=2) + 1e-8
    normalized = centered / scale

    return normalized.transpose(2, 0, 1).astype(np.float32)   # (3, T, J)
