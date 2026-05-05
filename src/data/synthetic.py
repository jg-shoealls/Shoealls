"""Synthetic data generation for algorithm validation."""

import numpy as np

# 11-class gait profiles
# 0:normal, 1:parkinson, 2:stroke, 3:diabetic_neuropathy, 4:cerebellar_ataxia,
# 5:osteoarthritis, 6:dementia, 7:cerebral_hemorrhage, 8:cerebral_infarction,
# 9:disc_herniation, 10:rheumatoid_arthritis
CLASS_NAMES = [
    "normal", "parkinson", "stroke", "diabetic_neuropathy", "cerebellar_ataxia",
    "osteoarthritis", "dementia", "cerebral_hemorrhage", "cerebral_infarction",
    "disc_herniation", "rheumatoid_arthritis",
]

_FREQ_MAP = {
    0: 1.8, 1: 1.0, 2: 1.1, 3: 1.3, 4: 1.5,
    5: 1.2, 6: 1.1, 7: 1.0, 8: 1.1, 9: 1.3, 10: 1.2,
}
_NOISE_MAP = {
    0: 0.10, 1: 0.15, 2: 0.25, 3: 0.30, 4: 0.40,
    5: 0.20, 6: 0.20, 7: 0.35, 8: 0.30, 9: 0.20, 10: 0.25,
}


def _generate_gait_cycle(
    num_frames: int,
    frequency: float,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
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
    freq = _FREQ_MAP[gait_class] + rng.uniform(-0.1, 0.1)
    noise = _NOISE_MAP[gait_class]
    t = np.linspace(0, num_frames / 30.0, num_frames)

    channels = []
    for _ in range(6):
        signal = _generate_gait_cycle(num_frames, freq, noise, rng)

        if gait_class == 1:   # Parkinson: tremor + reduced amplitude
            signal *= 0.5
            signal += 0.2 * np.sin(2 * np.pi * 5.0 * t)
        elif gait_class == 2:  # Stroke: hemiplegic asymmetry
            signal += 0.3 * np.sin(np.pi * freq * t)
        elif gait_class == 3:  # Diabetic neuropathy: steppage, high lift
            signal += 0.25 * np.abs(np.sin(np.pi * freq * t))
        elif gait_class == 4:  # Cerebellar ataxia: highly irregular
            signal += 0.35 * rng.standard_normal(num_frames)
        elif gait_class == 5:  # Osteoarthritis: antalgic, slow
            signal += 0.2 * np.sin(np.pi * freq * t) * 0.7
        elif gait_class == 6:  # Dementia: shuffling, reduced arm swing
            signal *= 0.6
        elif gait_class == 7:  # Cerebral hemorrhage: spastic asymmetry
            signal *= 0.55
            signal += 0.25 * rng.standard_normal(num_frames)
        elif gait_class == 8:  # Cerebral infarction: hemiparetic
            signal += 0.2 * np.sin(np.pi * freq * t)
            signal *= 0.7
        elif gait_class == 9:  # Disc herniation: trunk lean, antalgic
            signal += 0.15 * np.sin(np.pi * freq * t)
        elif gait_class == 10:  # Rheumatoid arthritis: stiff, slow
            signal *= 0.65
            signal += 0.1 * rng.standard_normal(num_frames)

        channels.append(signal)

    return np.stack(channels, axis=1).astype(np.float32)


def generate_synthetic_pressure(
    num_frames: int,
    gait_class: int,
    grid_size: tuple,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = grid_size
    freq = _FREQ_MAP[gait_class] + rng.uniform(-0.1, 0.1)
    t = np.linspace(0, num_frames / 30.0, num_frames)
    pressure = np.zeros((num_frames, h, w), dtype=np.float32)

    heel_region = slice(h // 2, h), slice(w // 4, 3 * w // 4)
    toe_region = slice(0, h // 3), slice(w // 4, 3 * w // 4)

    heel_signal = np.clip(np.sin(2 * np.pi * freq * t), 0, 1)
    toe_signal = np.clip(np.sin(2 * np.pi * freq * t + np.pi / 2), 0, 1)

    for i in range(num_frames):
        pressure[i][heel_region] = heel_signal[i]
        pressure[i][toe_region] = toe_signal[i]

    if gait_class == 1:    # Parkinson: flat-footed, shuffle
        pressure = np.clip(pressure, 0.2, 0.8)
    elif gait_class == 2:  # Stroke: unilateral loading
        pressure[:, :, :w // 2] *= 0.4
    elif gait_class == 3:  # Diabetic neuropathy: reduced sensation, forefoot
        pressure[:, h // 2:, :] *= 0.5
        pressure[:, :h // 3, :] *= 1.3
    elif gait_class == 4:  # Cerebellar ataxia: wide, irregular
        pressure += 0.3 * rng.standard_normal(pressure.shape).astype(np.float32)
    elif gait_class == 5:  # Osteoarthritis: uneven loading
        pressure[:, :, :w // 2] *= 0.6
    elif gait_class == 6:  # Dementia: flat, shuffling
        pressure = np.clip(pressure, 0.15, 0.75)
    elif gait_class == 7:  # Cerebral hemorrhage: spastic, asymmetric
        pressure[:, :, w // 2:] *= 0.3
        pressure += 0.2 * rng.standard_normal(pressure.shape).astype(np.float32)
    elif gait_class == 8:  # Cerebral infarction: hemiparetic loading
        pressure[:, :, :w // 2] *= 0.5
    elif gait_class == 9:  # Disc herniation: antalgic unloading
        pressure[:, :, w // 2:] *= 0.7
    elif gait_class == 10: # Rheumatoid arthritis: forefoot offloading
        pressure[:, h // 2:, :] *= 1.2
        pressure[:, :h // 3, :] *= 0.6

    noise = 0.05 * rng.standard_normal(pressure.shape).astype(np.float32)
    return np.clip(pressure + noise, 0, 1)


def generate_synthetic_skeleton(
    num_frames: int,
    gait_class: int,
    num_joints: int,
    rng: np.random.Generator,
) -> np.ndarray:
    freq = _FREQ_MAP[gait_class] + rng.uniform(-0.1, 0.1)
    t = np.linspace(0, num_frames / 30.0, num_frames)
    skeleton = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

    base_pose = np.zeros((num_joints, 3), dtype=np.float32)
    base_pose[0] = [0, 0.9, 0]
    base_pose[1] = [0, 1.5, 0]
    base_pose[2] = [0, 1.7, 0]
    base_pose[3] = [-0.2, 1.4, 0]
    base_pose[4] = [-0.4, 1.1, 0]
    base_pose[5] = [-0.4, 0.8, 0]
    base_pose[6] = [0.2, 1.4, 0]
    base_pose[7] = [0.4, 1.1, 0]
    base_pose[8] = [0.4, 0.8, 0]
    base_pose[9] = [-0.1, 0.9, 0]
    base_pose[10] = [-0.1, 0.5, 0]
    base_pose[11] = [-0.1, 0.05, 0]
    base_pose[12] = [0.1, 0.9, 0]
    base_pose[13] = [0.1, 0.5, 0]
    base_pose[14] = [0.1, 0.05, 0]
    if num_joints > 15:
        base_pose[15] = [-0.1, 0.0, 0.1]
        base_pose[16] = [0.1, 0.0, 0.1]

    for i in range(num_frames):
        skeleton[i] = base_pose.copy()
        phase = 2 * np.pi * freq * t[i]
        swing_amp = 0.15
        arm_amp = 0.08

        skeleton[i, 10, 0] += swing_amp * np.sin(phase)
        skeleton[i, 11, 0] += swing_amp * np.sin(phase) * 1.5
        skeleton[i, 13, 0] -= swing_amp * np.sin(phase)
        skeleton[i, 14, 0] -= swing_amp * np.sin(phase) * 1.5
        skeleton[i, 4, 0] -= arm_amp * np.sin(phase)
        skeleton[i, 5, 0] -= arm_amp * np.sin(phase) * 1.2
        skeleton[i, 7, 0] += arm_amp * np.sin(phase)
        skeleton[i, 8, 0] += arm_amp * np.sin(phase) * 1.2
        skeleton[i, :, 2] += 0.5 * t[i]

    if gait_class == 1:    # Parkinson: forward lean, reduced motion
        skeleton[:, 1:3, 2] -= 0.05
        skeleton[:, 3:9, :] *= 0.6
    elif gait_class == 2:  # Stroke: hemiplegic, unilateral arm reduction
        skeleton[:, 3:6, :] *= 0.4
        skeleton[:, 9:12, 0] -= 0.04
    elif gait_class == 3:  # Diabetic neuropathy: high-stepping
        skeleton[:, 10, 1] += 0.06
        skeleton[:, 13, 1] += 0.06
    elif gait_class == 4:  # Cerebellar ataxia: wide base, unsteady
        skeleton[:, 9:12, 0] -= 0.07
        skeleton[:, 12:15, 0] += 0.07
        skeleton += 0.02 * rng.standard_normal(skeleton.shape).astype(np.float32)
    elif gait_class == 5:  # Osteoarthritis: antalgic, limping
        skeleton[:, 10:12, 1] += 0.03
    elif gait_class == 6:  # Dementia: stooped, shuffling
        skeleton[:, 1:3, 2] -= 0.04
        skeleton[:, 3:9, :] *= 0.65
    elif gait_class == 7:  # Cerebral hemorrhage: spastic arm, asymmetric
        skeleton[:, 3:6, :] *= 0.3
        skeleton += 0.015 * rng.standard_normal(skeleton.shape).astype(np.float32)
    elif gait_class == 8:  # Cerebral infarction: hemiparetic gait
        skeleton[:, 3:6, :] *= 0.5
        skeleton[:, 10:12, 0] += 0.03
    elif gait_class == 9:  # Disc herniation: trunk lateral lean
        skeleton[:, 0:2, 0] += 0.04
        skeleton[:, 10:12, 1] += 0.02
    elif gait_class == 10: # Rheumatoid arthritis: stiff joints, slow
        skeleton *= 0.85
        skeleton += 0.008 * rng.standard_normal(skeleton.shape).astype(np.float32)

    noise = 0.005 * rng.standard_normal(skeleton.shape).astype(np.float32)
    return (skeleton + noise).astype(np.float32)


def generate_synthetic_dataset(
    num_samples_per_class: int = 50,
    num_frames: int = 150,
    num_classes: int = 4,
    grid_size: tuple = (16, 8),
    num_joints: int = 17,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    imu_list, pressure_list, skeleton_list, labels = [], [], [], []

    for class_idx in range(num_classes):
        for _ in range(num_samples_per_class):
            n = num_frames + rng.integers(-10, 10)
            imu_list.append(generate_synthetic_imu(n, class_idx, rng))
            pressure_list.append(generate_synthetic_pressure(n, class_idx, grid_size, rng))
            skeleton_list.append(generate_synthetic_skeleton(n, class_idx, num_joints, rng))
            labels.append(class_idx)

    return {
        "imu": imu_list,
        "pressure": pressure_list,
        "skeleton": skeleton_list,
        "labels": np.array(labels, dtype=np.int64),
        "class_names": CLASS_NAMES[:num_classes],
    }
