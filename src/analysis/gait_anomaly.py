"""비정상 보행 패턴 감지 엔진.

12가지 이상 보행 패턴을 감지하고 부상 위험과 연결:
  - 시공간 이상: 보폭 비대칭, 보행률 이상, 속도 저하
  - 압력 이상: 전족부 과부하, 뒤꿈치 회피, 외측/내측 편향
  - 동적 이상: COP 불안정, 체간 동요, 추진력 저하
  - 복합 패턴: 절뚝거림(antalgic), 회선 보행, 동결 보행

각 패턴에 대해:
  1. 규칙 기반 감지 (임상 기준)
  2. 심각도 스코어링 (0~1)
  3. 연관 부상 위험 매핑
  4. 한국어 임상 해석 + 교정 권고
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class AnomalyPattern:
    """감지된 비정상 보행 패턴."""
    pattern_id: str
    korean_name: str
    severity: float           # 0.0 ~ 1.0
    severity_label: str       # 정상/경미/주의/경고/위험
    description: str          # 한국어 설명
    evidence: list[str]       # 감지 근거
    injury_risks: list[str]   # 연관 부상 위험
    correction: str           # 교정 권고


@dataclass
class GaitAnomalyReport:
    """보행 이상 종합 보고서."""
    patterns: list[AnomalyPattern]          # 감지된 모든 패턴
    abnormal_patterns: list[AnomalyPattern] # 비정상으로 판정된 패턴만
    anomaly_score: float                    # 종합 이상 점수 (0~1)
    anomaly_grade: str                      # 정상/경미/주의/경고/위험
    injury_risk_summary: dict[str, float]   # 부상유형 → 종합 위험도
    summary_kr: str                         # 한국어 보고서


# ── 비정상 보행 패턴 정의 ─────────────────────────────────────────────
ANOMALY_DEFINITIONS = {
    # === 시공간 이상 ===
    "stride_asymmetry": {
        "korean_name": "보폭 비대칭",
        "category": "시공간",
        "description": "좌우 보폭 길이 또는 시간의 차이가 큰 패턴",
        "features": {
            "step_symmetry": {"direction": "low", "mild": 0.85, "severe": 0.65},
            "pressure_asymmetry": {"direction": "high", "mild": 0.10, "severe": 0.25},
        },
        "injury_risks": ["발목 염좌", "무릎 인대 손상", "고관절 통증", "요통"],
        "correction": "보행 시 좌우 걸음 간격을 의식적으로 맞추고, 약측 하지 근력 강화 운동(스쿼트, 런지) 실시",
    },
    "cadence_abnormal": {
        "korean_name": "보행률 이상",
        "category": "시공간",
        "description": "비정상적으로 높거나 낮은 분당 걸음수",
        "features": {
            "cadence_low": {"direction": "low", "mild": 95, "severe": 75},
            "cadence_high": {"direction": "high", "mild": 135, "severe": 160},
        },
        "injury_risks": ["족저근막염", "아킬레스건염", "경골 피로골절"],
        "correction": "적정 보행률(100~120 steps/min) 유지, 메트로놈 활용 보행 훈련",
    },
    "slow_gait": {
        "korean_name": "보행 속도 저하",
        "category": "시공간",
        "description": "정상 범위 이하의 보행 속도 (근력 저하, 통증 회피 의심)",
        "features": {
            "gait_speed": {"direction": "low", "mild": 1.0, "severe": 0.6},
        },
        "injury_risks": ["낙상", "근감소증", "관절 구축"],
        "correction": "점진적 보행 속도 증가 훈련, 하지 근력 강화, 유연성 운동 병행",
    },
    "irregular_stride": {
        "korean_name": "불규칙 보폭",
        "category": "시공간",
        "description": "보폭 간 규칙성이 낮은 패턴 (운동 조절 장애 의심)",
        "features": {
            "stride_regularity": {"direction": "low", "mild": 0.70, "severe": 0.45},
        },
        "injury_risks": ["낙상", "발목 염좌", "근골격계 과부하"],
        "correction": "리듬 보행 훈련, 균형 운동(한발 서기, 탠덤 보행), 시각적 보폭 가이드 사용",
    },
    # === 압력 분포 이상 ===
    "forefoot_overload": {
        "korean_name": "전족부 과부하",
        "category": "압력",
        "description": "앞발에 과도한 하중이 집중되는 패턴 (뒤꿈치 착지 부전)",
        "features": {
            "forefoot_pressure_ratio": {"direction": "high", "mild": 0.55, "severe": 0.72},
            "heel_pressure_ratio": {"direction": "low", "mild": 0.22, "severe": 0.12},
        },
        "injury_risks": ["중족골 피로골절", "중족골통", "족저근막염", "모턴 신경종"],
        "correction": "뒤꿈치 착지(heel strike) 연습, 스트레칭(종아리, 아킬레스건), 쿠셔닝 인솔 사용",
    },
    "heel_strike_excessive": {
        "korean_name": "과도한 뒤꿈치 충격",
        "category": "압력",
        "description": "뒤꿈치 착지 시 과도한 충격이 전달되는 패턴",
        "features": {
            "heel_pressure_ratio": {"direction": "high", "mild": 0.42, "severe": 0.58},
            "acceleration_rms": {"direction": "high", "mild": 2.5, "severe": 3.5},
        },
        "injury_risks": ["종골 피로골절", "족저근막염", "무릎 관절 충격", "경골 스트레스"],
        "correction": "중족부 착지(midfoot strike) 전환 훈련, 충격 흡수 인솔, 보행 속도 조절",
    },
    "lateral_deviation": {
        "korean_name": "외측 편향 보행",
        "category": "압력",
        "description": "체중이 발 외측으로 치우치는 과회외(supination) 패턴",
        "features": {
            "ml_variability": {"direction": "high", "mild": 0.10, "severe": 0.18},
        },
        "injury_risks": ["발목 내반 염좌", "제5 중족골 골절", "장경인대 증후군"],
        "correction": "내측 웨지 인솔 사용, 비골근(peroneal) 강화 운동, 불안정면 균형 훈련",
    },
    "medial_collapse": {
        "korean_name": "내측 붕괴 (과회내)",
        "category": "압력",
        "description": "아치가 무너지며 발 내측으로 하중이 집중되는 패턴",
        "features": {
            "arch_index": {"direction": "high", "mild": 0.35, "severe": 0.50},
        },
        "injury_risks": ["족저근막염", "후경골건 기능부전", "무릎 내반 변형", "평발 진행"],
        "correction": "아치 서포트 인솔 착용, 후경골건 강화 운동(타올 그랩, 카프 레이즈), 맨발 운동",
    },
    # === 동적 이상 ===
    "cop_instability": {
        "korean_name": "체중심 불안정",
        "category": "동적",
        "description": "체중심(COP) 궤적의 과도한 흔들림 → 균형 조절 장애",
        "features": {
            "cop_sway": {"direction": "high", "mild": 0.06, "severe": 0.12},
        },
        "injury_risks": ["낙상", "발목 염좌", "전신 근골격 과부하"],
        "correction": "균형 훈련(보수볼, 폼 패드), 고유감각 운동, 시각 피드백 보행 훈련",
    },
    "trunk_sway_excessive": {
        "korean_name": "과도한 체간 동요",
        "category": "동적",
        "description": "상체 좌우 흔들림이 과도한 패턴 (근력/균형 부족)",
        "features": {
            "trunk_sway": {"direction": "high", "mild": 3.0, "severe": 5.0},
        },
        "injury_risks": ["낙상", "요추 과부하", "고관절 통증"],
        "correction": "코어 안정화 운동(플랭크, 데드버그), 체간 균형 훈련, 힙 근력 강화",
    },
    "propulsion_deficit": {
        "korean_name": "추진력 저하",
        "category": "동적",
        "description": "보행 추진 단계에서 가속도가 약한 패턴",
        "features": {
            "acceleration_rms": {"direction": "low", "mild": 0.9, "severe": 0.5},
        },
        "injury_risks": ["근감소증 진행", "관절 구축", "보행 효율 저하"],
        "correction": "파워 워킹 훈련, 카프 레이즈, 발가락 푸시오프 강화, 경사면 보행",
    },
    # === 복합 패턴 ===
    "antalgic_gait": {
        "korean_name": "절뚝거림 (통증 회피 보행)",
        "category": "복합",
        "description": "통증을 피하기 위해 이환측 지지기를 단축하는 패턴",
        "features": {
            "step_symmetry": {"direction": "low", "mild": 0.82, "severe": 0.60},
            "gait_speed": {"direction": "low", "mild": 0.95, "severe": 0.65},
            "pressure_asymmetry": {"direction": "high", "mild": 0.12, "severe": 0.25},
        },
        "injury_risks": ["이차적 과부하 손상", "보상성 관절 통증", "고관절/무릎 변형"],
        "correction": "원인 통증 치료 우선, 점진적 체중 부하, 대칭 보행 재훈련, 보조기 검토",
    },
}

# ── 부상 위험 분류 ────────────────────────────────────────────────────
INJURY_CATEGORIES = {
    "족저근막염": {"severity_weight": 1.0, "timeline": "만성", "body_part": "발바닥"},
    "중족골 피로골절": {"severity_weight": 1.2, "timeline": "급성/만성", "body_part": "전족부"},
    "중족골통": {"severity_weight": 0.8, "timeline": "만성", "body_part": "전족부"},
    "모턴 신경종": {"severity_weight": 0.9, "timeline": "만성", "body_part": "전족부"},
    "발목 염좌": {"severity_weight": 1.1, "timeline": "급성", "body_part": "발목"},
    "발목 내반 염좌": {"severity_weight": 1.1, "timeline": "급성", "body_part": "발목"},
    "무릎 인대 손상": {"severity_weight": 1.3, "timeline": "급성", "body_part": "무릎"},
    "무릎 관절 충격": {"severity_weight": 0.9, "timeline": "만성", "body_part": "무릎"},
    "무릎 내반 변형": {"severity_weight": 1.0, "timeline": "만성", "body_part": "무릎"},
    "고관절 통증": {"severity_weight": 1.0, "timeline": "만성", "body_part": "고관절"},
    "고관절/무릎 변형": {"severity_weight": 1.1, "timeline": "만성", "body_part": "하지"},
    "요통": {"severity_weight": 1.0, "timeline": "만성", "body_part": "허리"},
    "요추 과부하": {"severity_weight": 1.0, "timeline": "만성", "body_part": "허리"},
    "종골 피로골절": {"severity_weight": 1.2, "timeline": "급성", "body_part": "뒤꿈치"},
    "아킬레스건염": {"severity_weight": 1.0, "timeline": "만성", "body_part": "발뒤꿈치"},
    "경골 피로골절": {"severity_weight": 1.2, "timeline": "급성", "body_part": "정강이"},
    "경골 스트레스": {"severity_weight": 0.9, "timeline": "만성", "body_part": "정강이"},
    "제5 중족골 골절": {"severity_weight": 1.3, "timeline": "급성", "body_part": "발 외측"},
    "장경인대 증후군": {"severity_weight": 0.9, "timeline": "만성", "body_part": "무릎 외측"},
    "후경골건 기능부전": {"severity_weight": 1.0, "timeline": "만성", "body_part": "발 내측"},
    "평발 진행": {"severity_weight": 0.7, "timeline": "만성", "body_part": "발"},
    "낙상": {"severity_weight": 1.5, "timeline": "급성", "body_part": "전신"},
    "근감소증": {"severity_weight": 0.8, "timeline": "만성", "body_part": "전신"},
    "근감소증 진행": {"severity_weight": 0.8, "timeline": "만성", "body_part": "전신"},
    "관절 구축": {"severity_weight": 0.9, "timeline": "만성", "body_part": "관절"},
    "근골격계 과부하": {"severity_weight": 0.9, "timeline": "만성", "body_part": "하지"},
    "전신 근골격 과부하": {"severity_weight": 0.9, "timeline": "만성", "body_part": "전신"},
    "이차적 과부하 손상": {"severity_weight": 1.0, "timeline": "만성", "body_part": "하지"},
    "보상성 관절 통증": {"severity_weight": 0.9, "timeline": "만성", "body_part": "하지"},
    "보행 효율 저하": {"severity_weight": 0.5, "timeline": "만성", "body_part": "전신"},
}


class GaitAnomalyDetector:
    """비정상 보행 패턴 감지 및 부상 위험 예측 엔진.

    보행 특성 데이터에서 12가지 이상 패턴을 감지하고,
    각 패턴에 연관된 부상 위험을 종합적으로 평가합니다.
    """

    def __init__(self):
        self.anomaly_defs = ANOMALY_DEFINITIONS
        self.injury_cats = INJURY_CATEGORIES

    def detect(self, features: dict[str, float]) -> GaitAnomalyReport:
        """보행 특성에서 비정상 패턴을 감지합니다.

        Args:
            features: 보행 바이오마커 딕셔너리.
                필수: gait_speed, cadence, stride_regularity, step_symmetry
                권장: cop_sway, arch_index, acceleration_rms, trunk_sway,
                      pressure_asymmetry, ml_variability,
                      heel_pressure_ratio, forefoot_pressure_ratio
        """
        # 파생 특성 계산
        enriched = dict(features)
        self._compute_derived(enriched)

        # 각 패턴 평가
        patterns = []
        for pattern_id, definition in self.anomaly_defs.items():
            pattern = self._evaluate_pattern(pattern_id, definition, enriched)
            patterns.append(pattern)

        # 비정상 패턴 필터
        abnormal = [p for p in patterns if p.severity > 0.0]
        abnormal.sort(key=lambda p: -p.severity)

        # 종합 이상 점수 (가중 RMS)
        if abnormal:
            scores = [p.severity for p in abnormal]
            anomaly_score = float(np.clip(np.sqrt(np.mean(np.square(scores))), 0, 1))
        else:
            anomaly_score = 0.0

        anomaly_grade = self._severity_label(anomaly_score)

        # 부상 위험 종합
        injury_risk_summary = self._aggregate_injury_risks(abnormal)

        # 보고서 생성
        summary = self._generate_report(patterns, abnormal, anomaly_score,
                                         anomaly_grade, injury_risk_summary)

        return GaitAnomalyReport(
            patterns=patterns,
            abnormal_patterns=abnormal,
            anomaly_score=round(anomaly_score, 3),
            anomaly_grade=anomaly_grade,
            injury_risk_summary=injury_risk_summary,
            summary_kr=summary,
        )

    def _compute_derived(self, features: dict):
        """파생 특성 자동 계산."""
        # cadence 분기 (높음/낮음)
        if "cadence" in features:
            features.setdefault("cadence_low", features["cadence"])
            features.setdefault("cadence_high", features["cadence"])

        # 압력 비율 계산 (zone features에서)
        heel_zones = ["zone_heel_medial_mean", "zone_heel_lateral_mean"]
        fore_zones = ["zone_forefoot_medial_mean", "zone_forefoot_lateral_mean", "zone_toes_mean"]
        mid_zones = ["zone_midfoot_medial_mean", "zone_midfoot_lateral_mean"]

        heel_sum = sum(features.get(z, 0) for z in heel_zones)
        fore_sum = sum(features.get(z, 0) for z in fore_zones)
        mid_sum = sum(features.get(z, 0) for z in mid_zones)
        total = heel_sum + fore_sum + mid_sum + 1e-8

        features.setdefault("heel_pressure_ratio", heel_sum / total)
        features.setdefault("forefoot_pressure_ratio", fore_sum / total)

        # 좌우 압력 비대칭
        if "ml_index" in features and "pressure_asymmetry" not in features:
            features["pressure_asymmetry"] = abs(features["ml_index"])

        # 좌우 흔들림 변동성
        if "cop_sway" in features and "ml_variability" not in features:
            features["ml_variability"] = features["cop_sway"] * 1.5

        # 체간 흔들림
        if "acceleration_rms" in features and "trunk_sway" not in features:
            features["trunk_sway"] = features["acceleration_rms"] * 1.2

    def _evaluate_pattern(
        self,
        pattern_id: str,
        definition: dict,
        features: dict,
    ) -> AnomalyPattern:
        """개별 패턴 평가."""
        feat_defs = definition["features"]
        scores = []
        evidence = []

        for feat_name, criteria in feat_defs.items():
            value = features.get(feat_name)
            if value is None:
                continue

            direction = criteria["direction"]
            mild_thresh = criteria["mild"]
            severe_thresh = criteria["severe"]

            if direction == "low":
                if value < mild_thresh:
                    # mild~severe 사이를 0~1로 매핑
                    range_size = mild_thresh - severe_thresh
                    if range_size > 0:
                        score = min((mild_thresh - value) / range_size, 1.0)
                    else:
                        score = 1.0
                    scores.append(score)
                    evidence.append(
                        f"{self._feat_korean(feat_name)}: {value:.3f} "
                        f"(정상 기준 {mild_thresh} 이상)"
                    )
                else:
                    scores.append(0.0)
            else:  # high
                if value > mild_thresh:
                    range_size = severe_thresh - mild_thresh
                    if range_size > 0:
                        score = min((value - mild_thresh) / range_size, 1.0)
                    else:
                        score = 1.0
                    scores.append(score)
                    evidence.append(
                        f"{self._feat_korean(feat_name)}: {value:.3f} "
                        f"(정상 기준 {mild_thresh} 이하)"
                    )
                else:
                    scores.append(0.0)

        # 종합 심각도 (최대값과 평균의 가중 평균)
        if scores and max(scores) > 0:
            max_score = max(scores)
            mean_score = np.mean([s for s in scores if s > 0])
            severity = float(np.clip(0.6 * max_score + 0.4 * mean_score, 0, 1))
        else:
            severity = 0.0

        severity_label = self._severity_label(severity)

        return AnomalyPattern(
            pattern_id=pattern_id,
            korean_name=definition["korean_name"],
            severity=round(severity, 3),
            severity_label=severity_label,
            description=definition["description"],
            evidence=evidence,
            injury_risks=definition["injury_risks"] if severity > 0 else [],
            correction=definition["correction"] if severity > 0 else "",
        )

    def _aggregate_injury_risks(
        self,
        abnormal_patterns: list[AnomalyPattern],
    ) -> dict[str, float]:
        """감지된 패턴에서 부상 위험을 종합합니다."""
        injury_scores: dict[str, list[float]] = {}

        for pattern in abnormal_patterns:
            for injury_name in pattern.injury_risks:
                weight = self.injury_cats.get(injury_name, {}).get("severity_weight", 1.0)
                weighted_score = pattern.severity * weight
                injury_scores.setdefault(injury_name, []).append(weighted_score)

        # 각 부상: 최대값과 평균의 조합
        result = {}
        for injury_name, scores in injury_scores.items():
            combined = 0.7 * max(scores) + 0.3 * np.mean(scores)
            result[injury_name] = round(float(np.clip(combined, 0, 1)), 3)

        # 위험도 순 정렬
        return dict(sorted(result.items(), key=lambda x: -x[1]))

    def _severity_label(self, score: float) -> str:
        """심각도 라벨."""
        if score >= 0.75:
            return "위험"
        elif score >= 0.50:
            return "경고"
        elif score >= 0.25:
            return "주의"
        elif score > 0.0:
            return "경미"
        else:
            return "정상"

    def _feat_korean(self, feat_name: str) -> str:
        """특성 이름 한국어 변환."""
        mapping = {
            "gait_speed": "보행 속도",
            "cadence": "보행률",
            "cadence_low": "보행률(저)",
            "cadence_high": "보행률(고)",
            "stride_regularity": "보폭 규칙성",
            "step_symmetry": "좌우 대칭성",
            "cop_sway": "체중심 흔들림",
            "ml_variability": "좌우 변동성",
            "heel_pressure_ratio": "뒤꿈치 하중",
            "forefoot_pressure_ratio": "앞발 하중",
            "arch_index": "아치 지수",
            "pressure_asymmetry": "압력 비대칭",
            "acceleration_rms": "가속도 크기",
            "trunk_sway": "체간 흔들림",
        }
        return mapping.get(feat_name, feat_name)

    def _generate_report(
        self,
        all_patterns: list[AnomalyPattern],
        abnormal: list[AnomalyPattern],
        anomaly_score: float,
        anomaly_grade: str,
        injury_risks: dict[str, float],
    ) -> str:
        """한국어 종합 보고서 생성."""
        lines = [
            "=" * 65,
            "  비정상 보행 패턴 감지 및 부상 위험 예측 보고서",
            "=" * 65,
            "",
        ]

        # 종합 점수
        lines.append(f"  종합 이상 점수: {anomaly_score:.0%} ({anomaly_grade})")
        lines.append(f"  감지된 이상 패턴: {len(abnormal)}개 / {len(all_patterns)}개 검사")
        lines.append("")

        # 전체 패턴 상태
        lines.append("─" * 65)
        lines.append("  [보행 패턴 검사 결과]")
        lines.append("")

        # 카테고리별 정리
        categories = {}
        for p in all_patterns:
            cat = self.anomaly_defs[p.pattern_id]["category"]
            categories.setdefault(cat, []).append(p)

        for cat_name in ["시공간", "압력", "동적", "복합"]:
            cat_patterns = categories.get(cat_name, [])
            if not cat_patterns:
                continue
            lines.append(f"  [{cat_name} 패턴]")
            for p in cat_patterns:
                if p.severity > 0:
                    bar_len = int(p.severity * 15)
                    bar = "█" * bar_len + "░" * (15 - bar_len)
                    marker = "▲"
                else:
                    bar = "░" * 15
                    marker = "○"
                lines.append(
                    f"  {marker} {p.korean_name:16s} [{bar}] "
                    f"{p.severity:.0%} ({p.severity_label})"
                )
            lines.append("")

        # 상세 이상 패턴
        if abnormal:
            lines.append("─" * 65)
            lines.append("  [감지된 이상 패턴 상세]")

            for p in abnormal[:5]:
                lines.append("")
                lines.append(f"  ■ {p.korean_name} (심각도: {p.severity:.0%}, {p.severity_label})")
                lines.append(f"    {p.description}")

                if p.evidence:
                    lines.append("    감지 근거:")
                    for ev in p.evidence:
                        lines.append(f"      - {ev}")

                if p.injury_risks:
                    lines.append(f"    연관 부상 위험: {', '.join(p.injury_risks[:4])}")

                lines.append(f"    교정 권고: {p.correction}")

        # 부상 위험 종합
        if injury_risks:
            lines.append("")
            lines.append("─" * 65)
            lines.append("  [부상 위험 종합 평가]")
            lines.append("")

            for i, (injury_name, risk_score) in enumerate(injury_risks.items()):
                if i >= 8:
                    lines.append(f"    ... 외 {len(injury_risks) - 8}개 항목")
                    break
                cat = self.injury_cats.get(injury_name, {})
                timeline = cat.get("timeline", "")
                body_part = cat.get("body_part", "")
                bar_len = int(risk_score * 15)
                bar = "█" * bar_len + "░" * (15 - bar_len)
                label = self._severity_label(risk_score)
                lines.append(
                    f"  {injury_name:16s} [{bar}] {risk_score:.0%} "
                    f"({label}) [{timeline}/{body_part}]"
                )

        # 권고사항
        lines.append("")
        lines.append("─" * 65)
        lines.append("  [권고사항]")
        lines.append("")

        if not abnormal:
            lines.append("  현재 보행 패턴에서 유의미한 이상 소견이 없습니다.")
            lines.append("  정기적인 보행 모니터링을 계속하시기 바랍니다.")
        else:
            high_risks = [name for name, score in injury_risks.items() if score >= 0.5]
            if high_risks:
                lines.append(f"  ※ 높은 부상 위험 감지: {', '.join(high_risks[:3])}")
                lines.append("  ※ 전문가 상담 및 정밀 검사를 권장합니다.")
            else:
                lines.append("  ※ 경미~주의 수준의 이상 패턴이 감지되었습니다.")
                lines.append("  ※ 교정 운동을 통해 보행 패턴 개선이 가능합니다.")

            # 교정 운동 요약
            corrections = []
            for p in abnormal[:3]:
                if p.correction:
                    corrections.append(f"  · {p.korean_name}: {p.correction}")
            if corrections:
                lines.append("")
                lines.append("  [우선 교정 운동]")
                lines.extend(corrections)

        lines.append("")
        lines.append("=" * 65)
        return "\n".join(lines)
