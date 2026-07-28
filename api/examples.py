"""샘플 데이터 생성 및 공용 예시 (테스트/문서용)."""

import numpy as np


# ── 정상 보행 특성 ────────────────────────────────────────────────────
NORMAL_GAIT_FEATURES = {
    "gait_speed": 1.25,
    "cadence": 118,
    "stride_regularity": 0.88,
    "step_symmetry": 0.93,
    "cop_sway": 0.035,
    "ml_index": 0.03,
    "arch_index": 0.24,
    "acceleration_rms": 1.6,
    "zone_heel_medial_mean": 0.32,
    "zone_heel_lateral_mean": 0.30,
    "zone_forefoot_medial_mean": 0.33,
    "zone_forefoot_lateral_mean": 0.30,
    "zone_toes_mean": 0.18,
    "zone_midfoot_medial_mean": 0.10,
    "zone_midfoot_lateral_mean": 0.08,
}

PARKINSONS_GAIT_FEATURES = {
    "gait_speed": 0.65,
    "cadence": 148,
    "stride_regularity": 0.42,
    "step_symmetry": 0.78,
    "cop_sway": 0.065,
    "ml_index": 0.08,
    "arch_index": 0.23,
    "acceleration_rms": 0.9,
    "zone_heel_medial_mean": 0.28,
    "zone_heel_lateral_mean": 0.25,
    "zone_forefoot_medial_mean": 0.38,
    "zone_forefoot_lateral_mean": 0.35,
    "zone_toes_mean": 0.12,
    "zone_midfoot_medial_mean": 0.10,
    "zone_midfoot_lateral_mean": 0.10,
}

STROKE_GAIT_FEATURES = {
    "gait_speed": 0.50,
    "cadence": 82,
    "stride_regularity": 0.52,
    "step_symmetry": 0.55,
    "cop_sway": 0.085,
    "ml_index": 0.22,
    "arch_index": 0.27,
    "acceleration_rms": 1.0,
    "zone_heel_medial_mean": 0.18,
    "zone_heel_lateral_mean": 0.35,
    "zone_forefoot_medial_mean": 0.42,
    "zone_forefoot_lateral_mean": 0.28,
    "zone_toes_mean": 0.10,
    "zone_midfoot_medial_mean": 0.08,
    "zone_midfoot_lateral_mean": 0.12,
}

FALL_RISK_FEATURES = {
    "gait_speed": 0.70,
    "cadence": 88,
    "stride_regularity": 0.55,
    "step_symmetry": 0.75,
    "cop_sway": 0.09,
    "ml_index": 0.15,
    "arch_index": 0.30,
    "acceleration_rms": 1.0,
    "zone_heel_medial_mean": 0.20,
    "zone_heel_lateral_mean": 0.28,
    "zone_forefoot_medial_mean": 0.36,
    "zone_forefoot_lateral_mean": 0.34,
    "zone_toes_mean": 0.15,
    "zone_midfoot_medial_mean": 0.10,
    "zone_midfoot_lateral_mean": 0.09,
}

GAIT_PROFILES = {
    "normal": NORMAL_GAIT_FEATURES,
    "parkinsons": PARKINSONS_GAIT_FEATURES,
    "stroke": STROKE_GAIT_FEATURES,
    "fall_risk": FALL_RISK_FEATURES,
}


def generate_sample_sensor_data(
    gait_class: int = 0,
    seq_len: int = 128,
    grid_h: int = 16,
    grid_w: int = 8,
    n_joints: int = 17,
    seed: int = 42,
) -> dict:
    """합성 센서 데이터 생성.

    Args:
        gait_class: 0=정상, 1=절뚝거림, 2=운동실조, 3=파킨슨
        seq_len: 시퀀스 길이 (기본 128)
        grid_h, grid_w: 압력 센서 그리드 크기
        n_joints: 스켈레톤 관절 수
        seed: 랜덤 시드

    Returns:
        {"imu": [...], "pressure": [...], "skeleton": [...]}
    """
    rng = np.random.default_rng(seed + gait_class)
    t = np.linspace(0, seq_len / 30.0, seq_len)

    freq_map = {0: 1.8, 1: 1.2, 2: 1.5, 3: 1.0}
    noise_map = {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.15}
    amp_map = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.5}

    freq = freq_map.get(gait_class, 1.8)
    noise = noise_map.get(gait_class, 0.1)
    amp = amp_map.get(gait_class, 1.0)

    # IMU: [seq_len, 6]
    imu = []
    for ch in range(6):
        base = amp * np.sin(2 * np.pi * freq * t + ch * np.pi / 6)
        if gait_class == 3:  # 파킨슨: 떨림
            base += 0.2 * np.sin(2 * np.pi * 5.0 * t)
        elif gait_class == 2:  # 운동실조: 불규칙
            base += 0.3 * rng.standard_normal(seq_len)
        base += noise * rng.standard_normal(seq_len)
        imu.append(base.tolist())
    imu_arr = list(map(list, zip(*imu)))  # [seq_len, 6]

    # Pressure: [grid_h, grid_w] — 정규화된 족저압 분포
    pressure = rng.random((grid_h, grid_w)) * 0.5 + 0.1
    if gait_class == 1:  # 절뚝거림: 비대칭
        pressure[:, :grid_w // 2] *= 0.5
    pressure = pressure.tolist()

    # Skeleton: [seq_len, n_joints, 3]
    skeleton = []
    for frame in range(seq_len):
        joints = []
        for j in range(n_joints):
            x = float(np.sin(2 * np.pi * freq * t[frame]) * 0.1 + j * 0.05)
            y = float(j * 0.1 + noise * rng.standard_normal())
            z = float(np.cos(2 * np.pi * freq * t[frame]) * 0.05)
            joints.append([x, y, z])
        skeleton.append(joints)

    return {"imu": imu_arr, "pressure": pressure, "skeleton": skeleton}
