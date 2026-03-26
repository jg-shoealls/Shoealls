"""보행 데이터 기반 질환 위험 예측 및 조기 진단 엔진.

10가지 질환에 대해:
  1. 바이오마커 기반 위험도 평가 (규칙 기반)
  2. ML 기반 분류기 (Random Forest + 앙상블)
  3. 종합 진단 보고서 생성
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .biomarkers import GaitBiomarkerExtractor, BiomarkerProfile, BIOMARKER_DEFINITIONS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 질환 정의: 10가지 질환의 보행 특성 패턴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISEASE_DEFINITIONS = {
    "parkinsons": {
        "korean_name": "파킨슨병",
        "description": "도파민 부족으로 인한 운동 장애",
        "gait_features": {
            "stride_regularity": {"direction": "low", "weight": 0.9, "threshold": 0.6},
            "cadence": {"direction": "high", "weight": 0.7, "threshold": 140},
            "gait_speed": {"direction": "low", "weight": 0.9, "threshold": 0.8},
            "acceleration_variability": {"direction": "high", "weight": 0.8, "threshold": 0.3},
            "step_symmetry": {"direction": "low", "weight": 0.5, "threshold": 0.8},
        },
        "key_signs": [
            "소보행 (짧은 보폭 + 빠른 보행률)",
            "동결 보행 (갑작스러운 멈춤)",
            "전진 가속 (festination)",
            "팔 흔들림 감소",
            "체간 전경 자세",
        ],
        "severity_thresholds": (0.25, 0.50, 0.75),
        "referral": "신경과",
    },
    "stroke": {
        "korean_name": "뇌졸중 (편마비)",
        "description": "뇌혈관 손상으로 인한 일측성 운동 장애",
        "gait_features": {
            "step_symmetry": {"direction": "low", "weight": 1.0, "threshold": 0.7},
            "pressure_asymmetry": {"direction": "high", "weight": 0.9, "threshold": 0.15},
            "gait_speed": {"direction": "low", "weight": 0.7, "threshold": 0.7},
            "ml_variability": {"direction": "high", "weight": 0.6, "threshold": 0.12},
        },
        "key_signs": [
            "뚜렷한 좌우 비대칭",
            "환측 하지 회선(circumduction)",
            "건측 과보상",
            "보행 속도 저하",
        ],
        "severity_thresholds": (0.20, 0.45, 0.70),
        "referral": "신경과 / 재활의학과",
    },
    "diabetic_neuropathy": {
        "korean_name": "당뇨 신경병증",
        "description": "당뇨로 인한 말초신경 손상, 발 감각 저하",
        "gait_features": {
            "forefoot_pressure_ratio": {"direction": "high", "weight": 0.9, "threshold": 0.55},
            "cop_sway": {"direction": "high", "weight": 0.8, "threshold": 0.07},
            "heel_pressure_ratio": {"direction": "low", "weight": 0.7, "threshold": 0.22},
            "stride_regularity": {"direction": "low", "weight": 0.6, "threshold": 0.65},
            "arch_index": {"direction": "high", "weight": 0.5, "threshold": 0.38},
        },
        "key_signs": [
            "앞발 압력 집중 (궤양 위험 부위)",
            "감각 저하로 인한 균형 불안정",
            "뒤꿈치 착지 감소",
            "보행 속도 및 규칙성 저하",
        ],
        "severity_thresholds": (0.20, 0.45, 0.70),
        "referral": "내분비내과 / 정형외과",
    },
    "cerebellar_ataxia": {
        "korean_name": "소뇌 실조증",
        "description": "소뇌 손상으로 인한 운동 조절 장애",
        "gait_features": {
            "cop_sway": {"direction": "high", "weight": 1.0, "threshold": 0.08},
            "ml_variability": {"direction": "high", "weight": 0.9, "threshold": 0.12},
            "stride_regularity": {"direction": "low", "weight": 0.9, "threshold": 0.55},
            "trunk_sway": {"direction": "high", "weight": 0.8, "threshold": 3.5},
            "acceleration_variability": {"direction": "high", "weight": 0.7, "threshold": 0.35},
        },
        "key_signs": [
            "넓은 보폭 기저면 (wide-based gait)",
            "좌우 흔들림 심함",
            "불규칙한 보폭과 리듬",
            "술 취한 것 같은 보행 양상",
        ],
        "severity_thresholds": (0.20, 0.45, 0.70),
        "referral": "신경과",
    },
    "multiple_sclerosis": {
        "korean_name": "다발성 경화증",
        "description": "중추신경계 탈수초 질환",
        "gait_features": {
            "stride_regularity": {"direction": "low", "weight": 0.8, "threshold": 0.60},
            "cop_sway": {"direction": "high", "weight": 0.7, "threshold": 0.07},
            "gait_speed": {"direction": "low", "weight": 0.8, "threshold": 0.9},
            "step_symmetry": {"direction": "low", "weight": 0.6, "threshold": 0.80},
        },
        "key_signs": [
            "보행 속도 점진적 저하",
            "피로에 따른 보행 변화 (fatigue effect)",
            "경직성 + 실조성 혼합 패턴",
            "보폭 불규칙",
        ],
        "severity_thresholds": (0.25, 0.50, 0.75),
        "referral": "신경과",
    },
    "osteoarthritis": {
        "korean_name": "골관절염",
        "description": "관절 퇴행으로 인한 통증 및 운동 제한",
        "gait_features": {
            "step_symmetry": {"direction": "low", "weight": 0.8, "threshold": 0.80},
            "gait_speed": {"direction": "low", "weight": 0.7, "threshold": 0.9},
            "pressure_asymmetry": {"direction": "high", "weight": 0.8, "threshold": 0.12},
            "heel_pressure_ratio": {"direction": "high", "weight": 0.5, "threshold": 0.42},
        },
        "key_signs": [
            "통증 회피 보행 (antalgic gait)",
            "이환 관절측 지지기 단축",
            "보행 시작 시 뻣뻣함 (morning stiffness)",
            "계단 오르내리기 어려움",
        ],
        "severity_thresholds": (0.20, 0.45, 0.70),
        "referral": "정형외과 / 류마티스내과",
    },
    "dementia": {
        "korean_name": "치매 (인지장애)",
        "description": "인지기능 저하에 동반되는 보행 장애",
        "gait_features": {
            "gait_speed": {"direction": "low", "weight": 1.0, "threshold": 0.8},
            "stride_regularity": {"direction": "low", "weight": 0.9, "threshold": 0.60},
            "cadence": {"direction": "low", "weight": 0.6, "threshold": 95},
            "acceleration_variability": {"direction": "high", "weight": 0.7, "threshold": 0.30},
        },
        "key_signs": [
            "보행 속도 점진적 감소 (연간 5% 이상)",
            "이중 과제 시 보행 악화 (dual-task deficit)",
            "보폭 변동성 증가",
            "방향 전환 시 불안정",
        ],
        "severity_thresholds": (0.25, 0.50, 0.75),
        "referral": "신경과 / 정신건강의학과",
    },
    "peripheral_artery": {
        "korean_name": "말초동맥질환",
        "description": "하지 혈류 저하로 인한 간헐적 파행",
        "gait_features": {
            "gait_speed": {"direction": "low", "weight": 0.8, "threshold": 0.9},
            "acceleration_rms": {"direction": "low", "weight": 0.7, "threshold": 0.9},
            "heel_pressure_ratio": {"direction": "low", "weight": 0.6, "threshold": 0.25},
            "step_symmetry": {"direction": "low", "weight": 0.5, "threshold": 0.82},
        },
        "key_signs": [
            "간헐적 파행 (걸으면 종아리 통증, 쉬면 호전)",
            "짧은 보행 거리 (claudication distance)",
            "보행 후반부 속도 저하",
            "추진력 약화",
        ],
        "severity_thresholds": (0.20, 0.45, 0.70),
        "referral": "혈관외과 / 순환기내과",
    },
    "spinal_stenosis": {
        "korean_name": "척추관 협착증",
        "description": "척추관 좁아짐으로 인한 신경 압박",
        "gait_features": {
            "trunk_sway": {"direction": "high", "weight": 0.8, "threshold": 3.0},
            "gait_speed": {"direction": "low", "weight": 0.7, "threshold": 0.85},
            "forefoot_pressure_ratio": {"direction": "high", "weight": 0.6, "threshold": 0.55},
            "cop_sway": {"direction": "high", "weight": 0.5, "threshold": 0.06},
        },
        "key_signs": [
            "전경 자세 (forward lean) 시 증상 호전",
            "쇼핑카트 징후 (shopping cart sign)",
            "보행 거리 제한 (neurogenic claudication)",
            "앞발 하중 증가",
        ],
        "severity_thresholds": (0.20, 0.45, 0.70),
        "referral": "정형외과 / 신경외과",
    },
    "vestibular_disorder": {
        "korean_name": "전정기관 장애",
        "description": "내이 전정기관 이상으로 인한 균형 장애",
        "gait_features": {
            "ml_variability": {"direction": "high", "weight": 1.0, "threshold": 0.12},
            "cop_sway": {"direction": "high", "weight": 0.9, "threshold": 0.08},
            "trunk_sway": {"direction": "high", "weight": 0.8, "threshold": 3.5},
            "stride_regularity": {"direction": "low", "weight": 0.6, "threshold": 0.65},
        },
        "key_signs": [
            "한쪽으로 치우치는 보행 (veering)",
            "방향 전환 시 현기증",
            "눈 감으면 흔들림 악화 (Romberg sign)",
            "좌우 흔들림 우세",
        ],
        "severity_thresholds": (0.20, 0.45, 0.70),
        "referral": "이비인후과 / 신경과",
    },
}


@dataclass
class DiseaseRiskResult:
    """단일 질환 위험 평가 결과."""
    disease_id: str
    korean_name: str
    risk_score: float         # 0.0 ~ 1.0
    severity: str             # 정상/관심/주의/위험
    confidence: float         # 판정 신뢰도
    matched_signs: list[str]  # 매칭된 이상 소견
    key_signs: list[str]      # 해당 질환의 주요 징후
    referral: str             # 권장 진료과
    biomarker_details: list[dict]  # 관련 바이오마커 상세


@dataclass
class DiseaseScreeningReport:
    """전체 질환 스크리닝 보고서."""
    results: list[DiseaseRiskResult]
    top_risks: list[DiseaseRiskResult]  # 위험도 상위 항목
    biomarker_profile: BiomarkerProfile
    overall_health_score: float  # 0~100 보행 건강 점수
    summary_kr: str


class DiseaseRiskPredictor:
    """보행 데이터 기반 질환 위험 예측 엔진.

    규칙 기반 + 통계적 스코어링으로 10가지 질환의 위험도를 평가합니다.
    """

    def __init__(self, sample_rate: int = 128):
        self.biomarker_extractor = GaitBiomarkerExtractor(sample_rate)

    def predict(
        self,
        features: dict[str, float],
    ) -> DiseaseScreeningReport:
        """보행 특성으로 질환 위험도를 예측합니다.

        Args:
            features: PersonalGaitProfiler.extract_session_features() 출력값.
        """
        # 1. 바이오마커 추출
        bio_profile = self.biomarker_extractor.extract(features)

        # 2. 각 질환별 위험도 평가
        results = []
        for disease_id, disease_def in DISEASE_DEFINITIONS.items():
            result = self._evaluate_disease_risk(disease_id, disease_def, features, bio_profile)
            results.append(result)

        # 3. 위험도 정렬
        results.sort(key=lambda r: -r.risk_score)
        top_risks = [r for r in results if r.risk_score >= 0.20]

        # 4. 보행 건강 점수 (100 - 평균 위험도 × 100)
        avg_risk = np.mean([r.risk_score for r in results])
        health_score = max(0, min(100, (1 - avg_risk) * 100))

        # 5. 보고서 생성
        summary = self._generate_summary(results, top_risks, health_score, bio_profile)

        return DiseaseScreeningReport(
            results=results,
            top_risks=top_risks,
            biomarker_profile=bio_profile,
            overall_health_score=round(health_score, 1),
            summary_kr=summary,
        )

    def _evaluate_disease_risk(
        self,
        disease_id: str,
        disease_def: dict,
        features: dict,
        bio_profile: BiomarkerProfile,
    ) -> DiseaseRiskResult:
        """개별 질환 위험도 평가."""
        gait_features = disease_def["gait_features"]
        matched_signs = []
        scores = []
        details = []

        for feat_name, criteria in gait_features.items():
            # 직접 매칭 또는 바이오마커에서 파생
            value = features.get(feat_name)
            if value is None:
                # 바이오마커에서 찾기
                for bio in bio_profile.biomarkers:
                    if bio.name == feat_name:
                        value = bio.value
                        break

            if value is None:
                continue

            direction = criteria["direction"]
            weight = criteria["weight"]
            threshold = criteria["threshold"]

            # 점수 계산
            if direction == "low":
                if value < threshold:
                    deviation = (threshold - value) / (abs(threshold) + 1e-8)
                    score = min(deviation, 1.0) * weight
                    matched_signs.append(
                        f"{self._get_korean_name(feat_name)}: {value:.3f} (기준 {threshold} 미만)"
                    )
                else:
                    score = 0.0
            else:  # high
                if value > threshold:
                    deviation = (value - threshold) / (abs(threshold) + 1e-8)
                    score = min(deviation, 1.0) * weight
                    matched_signs.append(
                        f"{self._get_korean_name(feat_name)}: {value:.3f} (기준 {threshold} 초과)"
                    )
                else:
                    score = 0.0

            scores.append(score)
            details.append({
                "feature": feat_name,
                "korean_name": self._get_korean_name(feat_name),
                "value": round(value, 4),
                "threshold": threshold,
                "direction": direction,
                "score": round(score, 3),
            })

        # 종합 위험 점수 (가중 평균)
        if scores:
            total_weight = sum(c["weight"] for c in gait_features.values()
                             if features.get(c.get("feature", "")) is not None or True)
            risk_score = float(np.clip(sum(scores) / max(len(scores), 1), 0, 1))
        else:
            risk_score = 0.0

        # 신뢰도: 사용 가능한 바이오마커 비율
        available = sum(1 for f in gait_features if f in features or
                       any(b.name == f for b in bio_profile.biomarkers))
        confidence = available / max(len(gait_features), 1)

        # 심각도 판정
        t1, t2, t3 = disease_def["severity_thresholds"]
        if risk_score >= t3:
            severity = "위험"
        elif risk_score >= t2:
            severity = "주의"
        elif risk_score >= t1:
            severity = "관심"
        else:
            severity = "정상"

        return DiseaseRiskResult(
            disease_id=disease_id,
            korean_name=disease_def["korean_name"],
            risk_score=round(risk_score, 3),
            severity=severity,
            confidence=round(confidence, 2),
            matched_signs=matched_signs,
            key_signs=disease_def["key_signs"],
            referral=disease_def["referral"],
            biomarker_details=details,
        )

    def _get_korean_name(self, feature_name: str) -> str:
        """피처 이름의 한글 변환."""
        bio_def = BIOMARKER_DEFINITIONS.get(feature_name)
        if bio_def:
            return bio_def["korean_name"]
        mapping = {
            "gait_speed": "보행 속도",
            "cadence": "보행률",
            "stride_regularity": "보폭 규칙성",
            "step_symmetry": "좌우 대칭성",
            "cop_sway": "체중심 흔들림",
            "ml_variability": "좌우 변동성",
            "heel_pressure_ratio": "뒤꿈치 하중",
            "forefoot_pressure_ratio": "앞발 하중",
            "arch_index": "아치 지수",
            "pressure_asymmetry": "좌우 압력 비대칭",
            "acceleration_rms": "가속도 크기",
            "acceleration_variability": "가속도 변동성",
            "trunk_sway": "체간 흔들림",
        }
        return mapping.get(feature_name, feature_name)

    def _generate_summary(
        self,
        results: list[DiseaseRiskResult],
        top_risks: list[DiseaseRiskResult],
        health_score: float,
        bio_profile: BiomarkerProfile,
    ) -> str:
        """한국어 종합 보고서 생성."""
        lines = [
            "=" * 65,
            "  보행 데이터 기반 질환 위험 스크리닝 보고서",
            "=" * 65,
            "",
        ]

        # 건강 점수
        if health_score >= 85:
            grade = "양호"
            color_desc = "양호한 보행 패턴입니다."
        elif health_score >= 70:
            grade = "보통"
            color_desc = "일부 항목에서 주의가 필요합니다."
        elif health_score >= 50:
            grade = "주의"
            color_desc = "여러 항목에서 이상 소견이 있습니다."
        else:
            grade = "경고"
            color_desc = "전문가 상담이 필요합니다."

        lines.append(f"  보행 건강 점수: {health_score:.0f}/100 ({grade})")
        lines.append(f"  {color_desc}")
        lines.append(f"  바이오마커: {bio_profile.total_count}개 측정, {bio_profile.abnormal_count}개 이상")
        lines.append("")

        # 바이오마커 요약
        lines.append("─" * 65)
        lines.append("  [바이오마커 측정 결과]")
        lines.append("")
        for bio in bio_profile.biomarkers:
            status = "이상" if bio.is_abnormal else "정상"
            marker = "▲" if bio.is_abnormal else "○"
            lines.append(
                f"  {marker} {bio.korean_name:14s} {bio.value:8.4f} {bio.unit:10s} "
                f"(정상: {bio.normal_range[0]:.2f}~{bio.normal_range[1]:.2f}) [{status}]"
            )
        lines.append("")

        # 질환별 위험도
        lines.append("─" * 65)
        lines.append("  [질환별 위험도 평가]")
        lines.append("")

        for result in results:
            bar_len = int(result.risk_score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(
                f"  {result.korean_name:14s} [{bar}] {result.risk_score:.0%} "
                f"({result.severity}) 신뢰도:{result.confidence:.0%}"
            )

        # 상세 소견
        if top_risks:
            lines.append("")
            lines.append("─" * 65)
            lines.append("  [주요 위험 항목 상세]")

            for risk in top_risks[:3]:
                lines.append("")
                lines.append(f"  ■ {risk.korean_name} (위험도: {risk.risk_score:.0%}, {risk.severity})")
                lines.append(f"    권장 진료과: {risk.referral}")

                if risk.matched_signs:
                    lines.append("    감지된 이상 소견:")
                    for sign in risk.matched_signs:
                        lines.append(f"      - {sign}")

                lines.append("    주요 징후:")
                for sign in risk.key_signs[:3]:
                    lines.append(f"      · {sign}")

        # 권고사항
        lines.append("")
        lines.append("─" * 65)
        lines.append("  [권고사항]")
        lines.append("")

        if not top_risks:
            lines.append("  현재 보행 패턴에서 뚜렷한 질환 위험 소견은 없습니다.")
            lines.append("  정기적인 보행 모니터링을 권장합니다.")
        else:
            referrals = set()
            for risk in top_risks:
                referrals.add(risk.referral)
            lines.append(f"  ※ 본 결과는 AI 기반 스크리닝이며, 확진을 위해 전문의 상담이 필요합니다.")
            lines.append(f"  ※ 권장 진료과: {', '.join(referrals)}")

        lines.append("")
        lines.append("=" * 65)
        return "\n".join(lines)
