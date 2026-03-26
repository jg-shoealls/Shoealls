"""보행 데이터 기반 질환 바이오마커 추출기.

보행 특성에서 각 질환의 임상 바이오마커를 추출하고,
의학 문헌 기반의 정상 범위와 비교하여 이상 여부를 판정합니다.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class BiomarkerResult:
    """단일 바이오마커 측정 결과."""
    name: str
    korean_name: str
    value: float
    normal_range: tuple[float, float]
    unit: str
    is_abnormal: bool
    deviation_pct: float  # 정상 범위 대비 이탈 정도 (%)
    clinical_meaning: str


@dataclass
class BiomarkerProfile:
    """전체 바이오마커 프로파일."""
    biomarkers: list[BiomarkerResult]
    abnormal_count: int
    total_count: int
    risk_categories: dict[str, list[BiomarkerResult]]  # disease -> related biomarkers


# ── 보행 바이오마커 정의 ──────────────────────────────────────────────
# 의학 문헌 기반 정상 범위 및 질환 연관성
BIOMARKER_DEFINITIONS = {
    # === 시공간 파라미터 ===
    "gait_speed": {
        "korean_name": "보행 속도",
        "unit": "m/s",
        "normal_range": (1.0, 1.4),
        "diseases": ["파킨슨병", "뇌졸중", "치매", "근감소증", "말초동맥질환", "뇌출혈", "뇌경색", "추간판 탈출증", "류마티스 관절염"],
        "meaning_low": "느린 보행 → 근력 저하, 신경계 이상 의심",
        "meaning_high": "과도한 보행 속도 → 충동 보행(festination) 의심",
    },
    "cadence": {
        "korean_name": "보행률 (분당 걸음수)",
        "unit": "steps/min",
        "normal_range": (100, 130),
        "diseases": ["파킨슨병", "치매", "근감소증"],
        "meaning_low": "낮은 보행률 → 운동 기능 저하",
        "meaning_high": "짧은 보폭 + 높은 보행률 → 파킨슨 특징",
    },
    "stride_regularity": {
        "korean_name": "보폭 규칙성",
        "unit": "ratio",
        "normal_range": (0.7, 1.0),
        "diseases": ["파킨슨병", "소뇌 실조증", "다발성경화증", "치매"],
        "meaning_low": "불규칙 보폭 → 운동 조절 장애 의심",
        "meaning_high": "",
    },
    "step_symmetry": {
        "korean_name": "좌우 대칭성",
        "unit": "ratio",
        "normal_range": (0.85, 1.0),
        "diseases": ["뇌졸중", "골관절염", "말초신경병증", "뇌출혈", "뇌경색", "추간판 탈출증", "류마티스 관절염"],
        "meaning_low": "좌우 비대칭 → 편마비, 관절 통증, 신경 손상 의심",
        "meaning_high": "",
    },
    # === 압력 분포 파라미터 ===
    "cop_sway": {
        "korean_name": "체중심 흔들림",
        "unit": "normalized",
        "normal_range": (0.0, 0.06),
        "diseases": ["소뇌 실조증", "전정기관 장애", "다발성경화증", "말초신경병증", "류마티스 관절염"],
        "meaning_low": "",
        "meaning_high": "과도한 흔들림 → 균형 장애, 낙상 위험",
    },
    "ml_variability": {
        "korean_name": "좌우 흔들림 변동성",
        "unit": "std",
        "normal_range": (0.0, 0.10),
        "diseases": ["소뇌 실조증", "전정기관 장애", "뇌졸중", "뇌출혈", "뇌경색"],
        "meaning_low": "",
        "meaning_high": "좌우 불안정 → 균형 조절 장애",
    },
    "heel_pressure_ratio": {
        "korean_name": "뒤꿈치 하중 비율",
        "unit": "ratio",
        "normal_range": (0.25, 0.40),
        "diseases": ["당뇨 신경병증", "말초동맥질환", "척추관협착증", "추간판 탈출증"],
        "meaning_low": "뒤꿈치 회피 → 통증, 궤양 위험",
        "meaning_high": "과도한 뒤꿈치 충격 → 족저근막염, 종골 스트레스",
    },
    "forefoot_pressure_ratio": {
        "korean_name": "앞발 하중 비율",
        "unit": "ratio",
        "normal_range": (0.35, 0.55),
        "diseases": ["당뇨 신경병증", "척추관협착증", "류마티스 관절염"],
        "meaning_low": "",
        "meaning_high": "앞발 과부하 → 당뇨족 궤양, 중족골 스트레스",
    },
    "arch_index": {
        "korean_name": "아치 지수",
        "unit": "ratio",
        "normal_range": (0.15, 0.35),
        "diseases": ["당뇨 신경병증", "류마티스 관절염", "샤르코-마리-투스병", "추간판 탈출증"],
        "meaning_low": "높은 아치 → 요족, 신경근육 질환 의심",
        "meaning_high": "낮은 아치(평발) → 과회내, 연부조직 퇴행",
    },
    "pressure_asymmetry": {
        "korean_name": "좌우 압력 비대칭",
        "unit": "index",
        "normal_range": (0.0, 0.12),
        "diseases": ["뇌졸중", "골관절염", "고관절 질환", "뇌출혈", "뇌경색", "추간판 탈출증", "류마티스 관절염"],
        "meaning_low": "",
        "meaning_high": "좌우 하중 차이 → 편마비, 관절 보호 보행",
    },
    # === IMU 기반 파라미터 ===
    "acceleration_rms": {
        "korean_name": "가속도 크기 (RMS)",
        "unit": "m/s²",
        "normal_range": (0.8, 2.5),
        "diseases": ["근감소증", "파킨슨병", "말초동맥질환"],
        "meaning_low": "약한 추진력 → 근력 저하, 동결 보행",
        "meaning_high": "과도한 충격 → 관절 부하 증가",
    },
    "acceleration_variability": {
        "korean_name": "가속도 변동성",
        "unit": "cv",
        "normal_range": (0.0, 0.35),
        "diseases": ["파킨슨병", "소뇌 실조증", "치매"],
        "meaning_low": "",
        "meaning_high": "불규칙한 움직임 → 운동 조절 장애",
    },
    "trunk_sway": {
        "korean_name": "체간 흔들림",
        "unit": "deg/s",
        "normal_range": (0.0, 3.0),
        "diseases": ["소뇌 실조증", "전정기관 장애", "파킨슨병", "척추관협착증", "뇌출혈", "추간판 탈출증"],
        "meaning_low": "",
        "meaning_high": "과도한 체간 동요 → 균형 장애, 낙상 고위험",
    },
}


class GaitBiomarkerExtractor:
    """보행 데이터에서 임상 바이오마커를 추출합니다.

    IMU + 족저압 데이터로부터 12개 바이오마커를 계산하고,
    각각의 정상 범위 대비 이탈 여부를 판정합니다.
    """

    def __init__(self, sample_rate: int = 128):
        self.sample_rate = sample_rate

    def extract(
        self,
        pressure_features: dict[str, float],
        imu_features: dict[str, float] | None = None,
    ) -> BiomarkerProfile:
        """보행 특성에서 바이오마커를 추출합니다.

        Args:
            pressure_features: FootZoneAnalyzer + PersonalGaitProfiler 출력값.
            imu_features: IMU 기반 특성 (옵션).

        Returns:
            BiomarkerProfile with all computed biomarkers.
        """
        biomarkers = []

        # 매핑: feature key -> biomarker key
        feature_mapping = {
            "cadence": "cadence",
            "stride_regularity": "stride_regularity",
            "step_symmetry": "step_symmetry",
            "cop_sway": "cop_sway",
            "arch_index": "arch_index",
            "acceleration_rms": "acceleration_rms",
        }

        all_features = dict(pressure_features)
        if imu_features:
            all_features.update(imu_features)

        # 추가 파생 바이오마커 계산
        self._compute_derived_features(all_features)

        for bio_key, bio_def in BIOMARKER_DEFINITIONS.items():
            # 해당 바이오마커에 대응하는 feature 찾기
            feat_key = feature_mapping.get(bio_key, bio_key)
            if feat_key not in all_features:
                continue

            value = all_features[feat_key]
            lo, hi = bio_def["normal_range"]

            # 이탈 판정
            if value < lo:
                is_abnormal = True
                deviation_pct = (lo - value) / (abs(lo) + 1e-8) * 100
                meaning = bio_def.get("meaning_low", "")
            elif value > hi:
                is_abnormal = True
                deviation_pct = (value - hi) / (abs(hi) + 1e-8) * 100
                meaning = bio_def.get("meaning_high", "")
            else:
                is_abnormal = False
                deviation_pct = 0.0
                meaning = "정상 범위 내"

            biomarkers.append(BiomarkerResult(
                name=bio_key,
                korean_name=bio_def["korean_name"],
                value=round(value, 4),
                normal_range=(lo, hi),
                unit=bio_def["unit"],
                is_abnormal=is_abnormal,
                deviation_pct=round(deviation_pct, 1),
                clinical_meaning=meaning if meaning else "정상 범위 내",
            ))

        # 질환별 관련 바이오마커 그룹핑
        risk_categories = {}
        for bio in biomarkers:
            bio_def = BIOMARKER_DEFINITIONS.get(bio.name, {})
            for disease in bio_def.get("diseases", []):
                risk_categories.setdefault(disease, []).append(bio)

        abnormal_count = sum(1 for b in biomarkers if b.is_abnormal)

        return BiomarkerProfile(
            biomarkers=biomarkers,
            abnormal_count=abnormal_count,
            total_count=len(biomarkers),
            risk_categories=risk_categories,
        )

    def _compute_derived_features(self, features: dict):
        """파생 바이오마커 계산."""
        # 보행 속도 추정 (보행률과 보폭에서)
        if "cadence" in features and "gait_speed" not in features:
            # 대략적 추정: speed ≈ cadence/60 * stride_length
            # stride_length ≈ 0.7~0.8m (평균), cadence in steps/min
            cadence = features["cadence"]
            stride_len = 0.75  # 기본값
            features["gait_speed"] = cadence / 60.0 * stride_len / 2.0

        # 좌우 압력 비대칭
        if "ml_index" in features and "pressure_asymmetry" not in features:
            features["pressure_asymmetry"] = abs(features["ml_index"])

        # 좌우 흔들림 변동성 (cop_sway에서 파생)
        if "cop_sway" in features and "ml_variability" not in features:
            features["ml_variability"] = features["cop_sway"] * 1.5

        # 가속도 변동성
        if "acceleration_rms" in features and "acceleration_variability" not in features:
            features["acceleration_variability"] = features.get("acceleration_rms", 1.0) * 0.2

        # 체간 흔들림 (가속도 기반 추정)
        if "acceleration_rms" in features and "trunk_sway" not in features:
            features["trunk_sway"] = features["acceleration_rms"] * 1.2

        # 뒤꿈치/앞발 하중 비율 (zone features에서 계산)
        heel_zones = ["zone_heel_medial_mean", "zone_heel_lateral_mean"]
        fore_zones = ["zone_forefoot_medial_mean", "zone_forefoot_lateral_mean", "zone_toes_mean"]
        mid_zones = ["zone_midfoot_medial_mean", "zone_midfoot_lateral_mean"]

        heel_sum = sum(features.get(z, 0) for z in heel_zones)
        fore_sum = sum(features.get(z, 0) for z in fore_zones)
        mid_sum = sum(features.get(z, 0) for z in mid_zones)
        total = heel_sum + fore_sum + mid_sum + 1e-8

        if "heel_pressure_ratio" not in features:
            features["heel_pressure_ratio"] = heel_sum / total
        if "forefoot_pressure_ratio" not in features:
            features["forefoot_pressure_ratio"] = fore_sum / total
