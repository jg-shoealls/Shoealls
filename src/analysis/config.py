"""분석 모듈 공통 설정 및 임상 임계값 중앙 관리.

모든 매직넘버, 정상 범위, 심각도 경계값을 단일 소스로 관리합니다.
"""

from __future__ import annotations

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 심각도 경계값
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEVERITY_THRESHOLDS_5LEVEL = {
    "위험": 0.75,
    "경고": 0.50,
    "주의": 0.25,
    "경미": 0.0,
}

SEVERITY_THRESHOLDS_4LEVEL = {
    "위험": 0.75,
    "경고": 0.50,
    "주의": 0.25,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 개인 프로파일 편차 임계값 (Z-score)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEVIATION_THRESHOLDS = {
    "mild": 1.5,      # σ
    "moderate": 2.0,   # σ
    "severe": 3.0,     # σ
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 추세 분석 임계값
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TREND_THRESHOLD = 0.05  # 기울기 유의 임계값

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 바이오마커 정상 범위 (의학문헌 기반)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BIOMARKER_NORMAL_RANGES: dict[str, tuple[float, float]] = {
    # 시공간 지표
    "gait_speed":             (1.0, 1.4),      # m/s
    "cadence":                (100.0, 130.0),   # steps/min
    "stride_regularity":      (0.7, 1.0),       # ratio
    "step_symmetry":          (0.85, 1.0),       # ratio
    # 압력 지표
    "heel_pressure_ratio":    (0.25, 0.40),     # ratio
    "forefoot_pressure_ratio": (0.35, 0.55),    # ratio
    "arch_index":             (0.15, 0.35),     # ratio
    "pressure_asymmetry":     (0.0, 0.12),      # index
    # 균형 지표
    "cop_sway":               (0.0, 0.06),      # normalized
    "ml_variability":         (0.0, 0.10),      # std
    "trunk_sway":             (0.0, 3.0),       # deg/s
    # IMU 지표
    "acceleration_rms":       (0.8, 2.5),       # m/s²
    "acceleration_variability": (0.0, 0.35),    # cv
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 부상 위험도 평가 정상 범위
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INJURY_NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "heel_pressure_ratio":    (0.25, 0.40),
    "forefoot_pressure_ratio": (0.35, 0.55),
    "arch_index":             (0.15, 0.35),
    "pressure_asymmetry":     (0.0, 0.12),
    "cop_sway":               (0.0, 0.06),
    "ml_variability":         (0.0, 0.10),
    "stride_regularity":      (0.7, 1.0),
    "step_symmetry":          (0.85, 1.0),
    "acceleration_rms":       (0.8, 2.5),
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 건강 점수 등급 경계값
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HEALTH_SCORE_GRADES = {
    "양호": 85,    # ≥ 85
    "보통": 70,    # ≥ 70
    "주의": 50,    # ≥ 50
    "경고": 0,     # < 50
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 센서 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRESSURE_GRID_SHAPE = (16, 8)   # rows × cols
IMU_CHANNELS = 6                # accel(3) + gyro(3)
SKELETON_JOINTS = 17            # COCO format
SKELETON_DIMS = 3               # x, y, z
SEQUENCE_LENGTH = 128           # 리샘플링 목표 길이

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 질환 카테고리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DISEASE_CATEGORIES = {
    "neurological": {
        "korean": "신경계",
        "diseases": [
            "parkinsons", "dementia", "cerebellar_ataxia",
            "multiple_sclerosis", "stroke", "cerebrovascular",
        ],
    },
    "musculoskeletal": {
        "korean": "근골격계",
        "diseases": [
            "osteoarthritis", "rheumatoid_arthritis",
            "disc_herniation", "spinal_stenosis",
        ],
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ML 분류기 기본 하이퍼파라미터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLASSIFIER_DEFAULTS = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "cv_folds": 5,
    "samples_per_class": 100,
    "random_state": 42,
}
