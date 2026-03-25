"""Synthetic data generation for algorithm validation."""

import numpy as np


def _generate_gait_cycle(
    num_frames: int,
    frequency: float,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a synthetic gait cycle signal."""
    t = np.linspace(0, num_frames / 30.0, num_frames)
    base = np.sin(2 * np.pi * frequency * t)
    harmonics = 0.3 * np.sin(4 * np.pi * frequency * t)
    noise = noise_level * rng.standard_normal(num_frames)
    return base + harmonics + noise


def generate_synthetic_imu(
    num_frames: int,
    gait_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic IMU data for a gait pattern.

    Args:
        num_frames: Number of time steps.
        gait_class: 0=normal, 1=antalgic, 2=ataxic, 3=parkinsonian.
        rng: Random number generator.

    Returns:
        IMU data of shape (num_frames, 6).
    """
    # Gait class determines frequency and regularity
    freq_map = {0: 1.8, 1: 1.2, 2: 1.5, 3: 1.0}
    noise_map = {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.15}

    freq = freq_map[gait_class] + rng.uniform(-0.1, 0.1)
    noise = noise_map[gait_class]

    channels = []
    for i in range(6):
        phase_offset = i * np.pi / 6
        t = np.linspace(0, num_frames / 30.0, num_frames)
        signal = _generate_gait_cycle(num_frames, freq, noise, rng)

        # Add class-specific characteristics
        if gait_class == 1:  # Antalgic: asymmetric pattern
            asymmetry = 0.3 * np.sin(np.pi * freq * t)
            signal += asymmetry
        elif gait_class == 2:  # Ataxic: irregular, wide-based
            irregular = 0.3 * rng.standard_normal(num_frames)
            signal += irregular
        elif gait_class == 3:  # Parkinsonian: reduced amplitude, shuffling
            signal *= 0.5
            tremor = 0.2 * np.sin(2 * np.pi * 5.0 * t)
            signal += tremor

        channels.append(signal)

    return np.stack(channels, axis=1).astype(np.float32)


def generate_synthetic_pressure(
    num_frames: int,
    gait_class: int,
    grid_size: tuple,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic plantar pressure data.

    Returns:
        Pressure data of shape (num_frames, H, W).
    """
    h, w = grid_size
    freq_map = {0: 1.8, 1: 1.2, 2: 1.5, 3: 1.0}
    freq = freq_map[gait_class] + rng.uniform(-0.1, 0.1)

    t = np.linspace(0, num_frames / 30.0, num_frames)
    pressure = np.zeros((num_frames, h, w), dtype=np.float32)

    # Heel and toe regions
    heel_region = slice(h // 2, h), slice(w // 4, 3 * w // 4)
    toe_region = slice(0, h // 3), slice(w // 4, 3 * w // 4)

    # Gait phase modulation
    heel_signal = np.clip(np.sin(2 * np.pi * freq * t), 0, 1)
    toe_signal = np.clip(np.sin(2 * np.pi * freq * t + np.pi / 2), 0, 1)

    for frame_idx in range(num_frames):
        pressure[frame_idx][heel_region] = heel_signal[frame_idx]
        pressure[frame_idx][toe_region] = toe_signal[frame_idx]

    # Class-specific modifications
    if gait_class == 1:  # Antalgic: uneven loading
        pressure[:, :, :w // 2] *= 0.5
    elif gait_class == 2:  # Ataxic: irregular pressure
        pressure += 0.3 * rng.standard_normal(pressure.shape).astype(np.float32)
    elif gait_class == 3:  # Parkinsonian: flat-footed
        pressure = np.clip(pressure, 0.2, 0.8)

    noise = 0.05 * rng.standard_normal(pressure.shape).astype(np.float32)
    return np.clip(pressure + noise, 0, 1)


def generate_synthetic_skeleton(
    num_frames: int,
    gait_class: int,
    num_joints: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic skeleton joint data.

    Returns:
        Skeleton data of shape (num_frames, num_joints, 3).
    """
    freq_map = {0: 1.8, 1: 1.2, 2: 1.5, 3: 1.0}
    freq = freq_map[gait_class] + rng.uniform(-0.1, 0.1)

    t = np.linspace(0, num_frames / 30.0, num_frames)
    skeleton = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

    # Base pose (standing) - simplified body model
    base_pose = np.zeros((num_joints, 3), dtype=np.float32)
    base_pose[0] = [0, 0.9, 0]      # hip center
    base_pose[1] = [0, 1.5, 0]      # spine
    base_pose[2] = [0, 1.7, 0]      # head
    base_pose[3] = [-0.2, 1.4, 0]   # left shoulder
    base_pose[4] = [-0.4, 1.1, 0]   # left elbow
    base_pose[5] = [-0.4, 0.8, 0]   # left wrist
    base_pose[6] = [0.2, 1.4, 0]    # right shoulder
    base_pose[7] = [0.4, 1.1, 0]    # right elbow
    base_pose[8] = [0.4, 0.8, 0]    # right wrist
    base_pose[9] = [-0.1, 0.9, 0]   # left hip
    base_pose[10] = [-0.1, 0.5, 0]  # left knee
    base_pose[11] = [-0.1, 0.05, 0] # left ankle
    base_pose[12] = [0.1, 0.9, 0]   # right hip
    base_pose[13] = [0.1, 0.5, 0]   # right knee
    base_pose[14] = [0.1, 0.05, 0]  # right ankle
    if num_joints > 15:
        base_pose[15] = [-0.1, 0.0, 0.1]  # left foot
        base_pose[16] = [0.1, 0.0, 0.1]   # right foot

    for frame_idx in range(num_frames):
        skeleton[frame_idx] = base_pose.copy()
        phase = 2 * np.pi * freq * t[frame_idx]

        # Leg swing
        swing_amp = 0.15
        skeleton[frame_idx, 10, 0] += swing_amp * np.sin(phase)       # left knee
        skeleton[frame_idx, 11, 0] += swing_amp * np.sin(phase) * 1.5 # left ankle
        skeleton[frame_idx, 13, 0] -= swing_amp * np.sin(phase)       # right knee
        skeleton[frame_idx, 14, 0] -= swing_amp * np.sin(phase) * 1.5 # right ankle

        # Arm swing (counter to legs)
        arm_amp = 0.08
        skeleton[frame_idx, 4, 0] -= arm_amp * np.sin(phase)
        skeleton[frame_idx, 5, 0] -= arm_amp * np.sin(phase) * 1.2
        skeleton[frame_idx, 7, 0] += arm_amp * np.sin(phase)
        skeleton[frame_idx, 8, 0] += arm_amp * np.sin(phase) * 1.2

        # Forward progression
        skeleton[frame_idx, :, 2] += 0.5 * t[frame_idx]

    # Class-specific modifications
    if gait_class == 1:  # Antalgic: limping, asymmetric
        skeleton[:, 10:12, 1] += 0.03  # left leg slightly elevated
    elif gait_class == 2:  # Ataxic: wide base, unsteady
        skeleton[:, 9:12, 0] -= 0.05   # wider left leg
        skeleton[:, 12:15, 0] += 0.05  # wider right leg
        skeleton += 0.02 * rng.standard_normal(skeleton.shape).astype(np.float32)
    elif gait_class == 3:  # Parkinsonian: reduced motion, forward lean
        skeleton[:, 1:3, 2] -= 0.05    # forward lean
        skeleton[:, 3:9, :] *= 0.7     # reduced arm swing

    noise = 0.005 * rng.standard_normal(skeleton.shape).astype(np.float32)
    return skeleton + noise


def generate_synthetic_dataset(
    num_samples_per_class: int = 50,
    num_frames: int = 150,
    num_classes: int = 4,
    grid_size: tuple = (16, 8),
    num_joints: int = 17,
    seed: int = 42,
) -> dict:
    """Generate a complete synthetic multimodal gait dataset.

    Returns:
        Dictionary with keys: 'imu', 'pressure', 'skeleton', 'labels'.
    """
    rng = np.random.default_rng(seed)
    total = num_samples_per_class * num_classes

    imu_list, pressure_list, skeleton_list, labels = [], [], [], []

    for class_idx in range(num_classes):
        for _ in range(num_samples_per_class):
            n = num_frames + rng.integers(-10, 10)
            imu_list.append(generate_synthetic_imu(n, class_idx, rng))
            pressure_list.append(
                generate_synthetic_pressure(n, class_idx, grid_size, rng)
            )
            skeleton_list.append(
                generate_synthetic_skeleton(n, class_idx, num_joints, rng)
            )
            labels.append(class_idx)

    return {
        "imu": imu_list,
        "pressure": pressure_list,
        "skeleton": skeleton_list,
        "labels": np.array(labels, dtype=np.int64),
        "class_names": ["normal", "antalgic", "ataxic", "parkinsonian"],
    }
