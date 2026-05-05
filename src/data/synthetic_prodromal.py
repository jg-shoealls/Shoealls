"""질환 발병 전(전임상~초기징후) 4단계 보행 데이터 합성 생성기.

파킨슨 진행 기준 4단계:
  0: 정상 (Normal)       — 건강한 보행
  1: 전임상 (Pre-clinical) — AI만 감지 가능한 미세 변화
  2: 초기 (Early-stage)   — 경미하지만 명확한 초기 징후
  3: 임상 (Clinical)      — 확립된 질환 패턴 (파킨슨)

센서 모달리티:
  - IMU (6채널)            : 가속도 3축 + 자이로 3축
  - 족저압 (16×8 그리드)  : 발바닥 압력 분포
  - 스켈레톤 (17관절 3D)  : 관절 위치
  - 지자기 + 기압 (4채널) : mx, my, mz (FOG 감지) + 기압고도 (발 지상고)

핵심 변화 지표:
  - 보행 주파수 감소      : 1.80 → 1.65 → 1.38 → 1.02 Hz
  - 안정시 떨림 증가      : 0 → 0.025 → 0.09 → 0.22 (normalized)
  - 좌우 비대칭           : 0 → 5% → 15% → 30%
  - 팔 흔들림 감소        : 100% → 90% → 72% → 45%
  - 보폭 규칙성           : 0.95 → 0.88 → 0.72 → 0.52
  - 보행 동결 (FOG)       : 0 → 0.03 → 0.10 → 0.25
  - 발 지상고             : 100% → 88% → 65% → 38%
"""

import numpy as np

STAGE_PROFILES = {
    0: {
        "name": "정상", "name_en": "Normal",
        "freq": 1.80, "freq_jitter": 0.08,
        "noise": 0.10,
        "amplitude": 1.00,
        "tremor_amp": 0.000, "tremor_freq": 5.0,
        "asymmetry": 0.00,
        "arm_swing_scale": 1.00,
        "stride_regularity": 0.95,
        "forward_lean": 0.000,
        "pressure_flatness": 0.00,
        # 신규: 지자기 + 기압
        "fog_amp": 0.000,          # 보행 동결 진폭 (magnetometer 진동)
        "heading_drift": 0.002,    # 방위각 편향 (rad/frame)
        "foot_clearance": 1.00,    # 발 지상고 (정규화)
        "clearance_cv": 0.06,      # 지상고 변동계수 (CoV)
    },
    1: {
        "name": "전임상", "name_en": "Pre-clinical",
        "freq": 1.65, "freq_jitter": 0.10,
        "noise": 0.12,
        "amplitude": 0.93,
        "tremor_amp": 0.025, "tremor_freq": 4.8,
        "asymmetry": 0.05,
        "arm_swing_scale": 0.90,
        "stride_regularity": 0.88,
        "forward_lean": 0.010,
        "pressure_flatness": 0.05,
        "fog_amp": 0.030,
        "heading_drift": 0.006,
        "foot_clearance": 0.88,
        "clearance_cv": 0.12,
    },
    2: {
        "name": "초기", "name_en": "Early-stage",
        "freq": 1.38, "freq_jitter": 0.12,
        "noise": 0.14,
        "amplitude": 0.78,
        "tremor_amp": 0.090, "tremor_freq": 5.1,
        "asymmetry": 0.15,
        "arm_swing_scale": 0.72,
        "stride_regularity": 0.72,
        "forward_lean": 0.025,
        "pressure_flatness": 0.20,
        "fog_amp": 0.100,
        "heading_drift": 0.014,
        "foot_clearance": 0.65,
        "clearance_cv": 0.22,
    },
    3: {
        "name": "임상", "name_en": "Clinical",
        "freq": 1.02, "freq_jitter": 0.08,
        "noise": 0.16,
        "amplitude": 0.52,
        "tremor_amp": 0.220, "tremor_freq": 5.2,
        "asymmetry": 0.30,
        "arm_swing_scale": 0.45,
        "stride_regularity": 0.52,
        "forward_lean": 0.060,
        "pressure_flatness": 0.45,
        "fog_amp": 0.250,
        "heading_drift": 0.028,
        "foot_clearance": 0.38,
        "clearance_cv": 0.38,
    },
}

CLASS_NAMES    = ["정상", "전임상", "초기", "임상"]
CLASS_NAMES_EN = ["Normal", "Pre-clinical", "Early-stage", "Clinical"]


def _base_signal(num_frames: int, freq: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, num_frames / 30.0, num_frames)
    return np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t) + noise * rng.standard_normal(num_frames)


# ── IMU ───────────────────────────────────────────────────────────────────────

def generate_prodromal_imu(num_frames: int, stage: int, rng: np.random.Generator) -> np.ndarray:
    """IMU 데이터 생성 — (num_frames, 6) float32."""
    p = STAGE_PROFILES[stage]
    freq = p["freq"] + rng.uniform(-p["freq_jitter"], p["freq_jitter"])
    t = np.linspace(0, num_frames / 30.0, num_frames)

    channels = []
    for ch in range(6):
        sig = _base_signal(num_frames, freq, p["noise"], rng) * p["amplitude"]

        if p["tremor_amp"] > 0:
            stance = np.clip(np.sin(2 * np.pi * freq * t), 0, 1)
            sig += p["tremor_amp"] * np.sin(2 * np.pi * p["tremor_freq"] * t) * (1 + 0.5 * stance)

        if p["asymmetry"] > 0 and ch % 2 == 0:
            sig *= (1 - p["asymmetry"] * rng.uniform(0.6, 1.4))

        if p["stride_regularity"] < 0.90:
            irregularity = (1 - p["stride_regularity"])
            phase_noise = np.cumsum(irregularity * rng.standard_normal(num_frames)) * 0.005
            sig = 0.8 * sig + 0.2 * np.sin(2 * np.pi * freq * t + phase_noise) * p["amplitude"]

        channels.append(sig)

    return np.stack(channels, axis=1).astype(np.float32)


# ── 족저압 ────────────────────────────────────────────────────────────────────

def generate_prodromal_pressure(num_frames: int, stage: int, grid_size: tuple, rng: np.random.Generator) -> np.ndarray:
    """족저압 데이터 생성 — (num_frames, H, W) float32."""
    p = STAGE_PROFILES[stage]
    h, w = grid_size
    freq = p["freq"] + rng.uniform(-p["freq_jitter"], p["freq_jitter"])
    t = np.linspace(0, num_frames / 30.0, num_frames)
    pressure = np.zeros((num_frames, h, w), dtype=np.float32)

    heel = (slice(h // 2, h), slice(w // 4, 3 * w // 4))
    toe  = (slice(0, h // 3), slice(w // 4, 3 * w // 4))

    heel_sig = np.clip(np.sin(2 * np.pi * freq * t), 0, 1) * p["amplitude"]
    toe_sig  = np.clip(np.sin(2 * np.pi * freq * t + np.pi / 2), 0, 1) * p["amplitude"]

    for i in range(num_frames):
        pressure[i][heel] = heel_sig[i]
        pressure[i][toe]  = toe_sig[i]

    if p["pressure_flatness"] > 0:
        flat = np.full((h, w), p["pressure_flatness"] * 0.4, dtype=np.float32)
        for i in range(num_frames):
            pressure[i] = pressure[i] * (1 - p["pressure_flatness"]) + flat

    if p["asymmetry"] > 0:
        pressure[:, :, :w // 2] *= (1 - p["asymmetry"] * 0.5)

    noise = 0.04 * rng.standard_normal(pressure.shape).astype(np.float32)
    return np.clip(pressure + noise, 0, 1)


# ── 스켈레톤 ──────────────────────────────────────────────────────────────────

def generate_prodromal_skeleton(num_frames: int, stage: int, num_joints: int, rng: np.random.Generator) -> np.ndarray:
    """스켈레톤 관절 데이터 생성 — (num_frames, num_joints, 3) float32."""
    p = STAGE_PROFILES[stage]
    freq = p["freq"] + rng.uniform(-p["freq_jitter"], p["freq_jitter"])
    t = np.linspace(0, num_frames / 30.0, num_frames)

    base = np.zeros((num_joints, 3), dtype=np.float32)
    base[0] = [0, 0.9, 0];   base[1] = [0, 1.5, 0];    base[2] = [0, 1.7, 0]
    base[3] = [-0.2, 1.4, 0]; base[4] = [-0.4, 1.1, 0]; base[5] = [-0.4, 0.8, 0]
    base[6] = [0.2, 1.4, 0];  base[7] = [0.4, 1.1, 0];  base[8] = [0.4, 0.8, 0]
    base[9] = [-0.1, 0.9, 0]; base[10] = [-0.1, 0.5, 0]; base[11] = [-0.1, 0.05, 0]
    base[12] = [0.1, 0.9, 0]; base[13] = [0.1, 0.5, 0];  base[14] = [0.1, 0.05, 0]
    if num_joints > 15:
        base[15] = [-0.1, 0.0, 0.1]; base[16] = [0.1, 0.0, 0.1]

    swing    = 0.15 * p["amplitude"]
    arm      = 0.08 * p["arm_swing_scale"]
    skeleton = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

    for i in range(num_frames):
        skeleton[i] = base.copy()
        ph = 2 * np.pi * freq * t[i]

        skeleton[i, 10, 0] += swing * np.sin(ph)
        skeleton[i, 11, 0] += swing * np.sin(ph) * 1.5
        skeleton[i, 13, 0] -= swing * np.sin(ph)
        skeleton[i, 14, 0] -= swing * np.sin(ph) * 1.5

        skeleton[i, 4, 0] -= arm * np.sin(ph)
        skeleton[i, 5, 0] -= arm * np.sin(ph) * 1.2
        skeleton[i, 7, 0] += arm * np.sin(ph)
        skeleton[i, 8, 0] += arm * np.sin(ph) * 1.2

        skeleton[i, :, 2] += 0.5 * t[i]

    if p["forward_lean"] > 0:
        skeleton[:, 1:3, 2] -= p["forward_lean"]

    if p["asymmetry"] > 0:
        skeleton[:, 9:12, 0] *= (1 - p["asymmetry"] * 0.4)

    noise = 0.005 * rng.standard_normal(skeleton.shape).astype(np.float32)
    return (skeleton + noise).astype(np.float32)


# ── 지자기 센서 (Magnetometer) ────────────────────────────────────────────────

def generate_prodromal_magnetometer(num_frames: int, stage: int, rng: np.random.Generator) -> np.ndarray:
    """지자기 데이터 생성 — (num_frames, 3) float32.

    FOG(보행 동결) 발생 시 3~8Hz 대역에서 고주파 진동이 나타난다.
    헤딩 드리프트: 파킨슨 진행 시 직선 보행 능력 저하.
    """
    p = STAGE_PROFILES[stage]
    t = np.linspace(0, num_frames / 30.0, num_frames)

    # 기저 헤딩: 천천히 회전하는 방위각
    heading = np.cumsum(p["heading_drift"] * (1 + 0.3 * rng.standard_normal(num_frames)))

    # FOG 진동: 3~8Hz 랜덤 주파수의 고주파 진동
    if p["fog_amp"] > 0:
        fog_freq = rng.uniform(3.5, 7.5)
        fog_phase = rng.uniform(0, 2 * np.pi)
        # FOG 에피소드: 랜덤 구간에 집중 발생
        fog_envelope = np.ones(num_frames)
        if stage >= 2:
            n_episodes = rng.integers(1, 4)
            for _ in range(n_episodes):
                start = rng.integers(0, num_frames - num_frames // 5)
                length = rng.integers(num_frames // 10, num_frames // 4)
                fog_envelope[start:start + length] *= rng.uniform(1.5, 3.0)
        heading += p["fog_amp"] * fog_envelope * np.sin(2 * np.pi * fog_freq * t + fog_phase)

    # 지자기 벡터 (수평 성분: mx, my / 수직 성분: mz)
    mx = np.cos(heading).astype(np.float32) + (p["noise"] * 0.3 * rng.standard_normal(num_frames)).astype(np.float32)
    my = np.sin(heading).astype(np.float32) + (p["noise"] * 0.3 * rng.standard_normal(num_frames)).astype(np.float32)
    mz = (np.full(num_frames, 0.45) + p["noise"] * 0.1 * rng.standard_normal(num_frames)).astype(np.float32)

    return np.stack([mx, my, mz], axis=1).astype(np.float32)


# ── 기압 센서 (Barometer — 발 지상고 프록시) ─────────────────────────────────

def generate_prodromal_barometer(num_frames: int, stage: int, rng: np.random.Generator) -> np.ndarray:
    """기압 고도 데이터 생성 — (num_frames, 1) float32.

    유각기(swing phase)에서 발이 올라가면 기압 고도가 증가한다.
    파킨슨 진행 시: 발 들어올림 높이 감소 + 변동성 증가 (shuffle gait).
    """
    p = STAGE_PROFILES[stage]
    freq = p["freq"] + rng.uniform(-p["freq_jitter"], p["freq_jitter"])
    t = np.linspace(0, num_frames / 30.0, num_frames)

    # 보행 주기에 맞춘 발 지상고 — 유각기에 피크
    swing_phase = np.clip(np.sin(2 * np.pi * freq * t + np.pi * 0.3), 0, 1)

    # 걸음마다 지상고 편차 (stride variability)
    n_strides = max(1, int(freq * num_frames / 30.0))
    per_stride_scale = 1.0 + p["clearance_cv"] * rng.standard_normal(num_frames)
    per_stride_scale = np.clip(per_stride_scale, 0.2, 2.0)

    altitude = p["foot_clearance"] * swing_phase * per_stride_scale

    # 비대칭: 한쪽 발 지상고 감소 (편측 파킨슨)
    if p["asymmetry"] > 0.1:
        # 홀수 보폭 주기에서 감소
        asymm_mask = (np.sin(2 * np.pi * freq * t) > 0).astype(np.float32)
        altitude *= 1 - p["asymmetry"] * 0.4 * asymm_mask

    noise = p["noise"] * 0.05 * rng.standard_normal(num_frames).astype(np.float32)
    return np.clip(altitude + noise, 0, None).astype(np.float32)[:, np.newaxis]


# ── 전체 데이터셋 생성 ────────────────────────────────────────────────────────

def generate_prodromal_dataset(
    num_samples_per_stage: int = 100,
    num_frames: int = 128,
    grid_size: tuple = (16, 8),
    num_joints: int = 17,
    seed: int = 42,
) -> dict:
    """4단계 질환 진행 멀티모달 데이터셋 생성.

    Returns:
        dict: imu, pressure, skeleton, magnetometer, barometer, labels, class_names, stage_profiles
    """
    rng = np.random.default_rng(seed)
    imu_list, pressure_list, skeleton_list = [], [], []
    mag_list, baro_list, labels = [], [], []

    for stage in range(4):
        for _ in range(num_samples_per_stage):
            n = num_frames + rng.integers(-10, 10)
            imu_list.append(generate_prodromal_imu(n, stage, rng))
            pressure_list.append(generate_prodromal_pressure(n, stage, grid_size, rng))
            skeleton_list.append(generate_prodromal_skeleton(n, stage, num_joints, rng))
            mag_list.append(generate_prodromal_magnetometer(n, stage, rng))
            baro_list.append(generate_prodromal_barometer(n, stage, rng))
            labels.append(stage)

    return {
        "imu":          imu_list,
        "pressure":     pressure_list,
        "skeleton":     skeleton_list,
        "magnetometer": mag_list,
        "barometer":    baro_list,
        "labels":       np.array(labels, dtype=np.int64),
        "class_names":    CLASS_NAMES,
        "class_names_en": CLASS_NAMES_EN,
        "stage_profiles": STAGE_PROFILES,
    }
