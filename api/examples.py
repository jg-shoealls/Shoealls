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
    """학습 데이터 생성기와 동일한 패턴으로 합성 센서 데이터 생성.

    Args:
        gait_class: 0=정상, 1=절뚝거림, 2=운동실조, 3=파킨슨
        seq_len: 시퀀스 길이 (기본 128)
        grid_h, grid_w: 압력 센서 그리드 크기
        n_joints: 스켈레톤 관절 수
        seed: 랜덤 시드

    Returns:
        {"imu": [...], "pressure": [...], "skeleton": [...]}
    """
    from src.data.synthetic import (
        generate_synthetic_imu,
        generate_synthetic_pressure,
        generate_synthetic_skeleton,
    )

    rng = np.random.default_rng(seed + gait_class)

    # IMU: (seq_len, 6)
    imu_arr = generate_synthetic_imu(seq_len, gait_class, rng)

    # Pressure: (seq_len, H, W) → API는 단일 프레임 (H, W) 평균 사용
    pressure_seq = generate_synthetic_pressure(seq_len, gait_class, (grid_h, grid_w), rng)
    pressure = pressure_seq.mean(axis=0)  # (H, W)

    # Skeleton: (seq_len, n_joints, 3)
    skeleton_arr = generate_synthetic_skeleton(seq_len, gait_class, n_joints, rng)

    return {
        "imu":      imu_arr.tolist(),           # [[float*6] * seq_len]
        "pressure": pressure.tolist(),           # [[float*W] * H]
        "skeleton": skeleton_arr.tolist(),       # [[[float*3] * n_joints] * seq_len]
    }
