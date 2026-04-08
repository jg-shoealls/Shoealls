"""파킨슨병 특화 보행 분석 엔진.

파킨슨병의 5대 보행 하위 패턴을 개별 감지하고,
Hoehn & Yahr 단계 추정, 종합 위험도, 진료 권고를 생성합니다.

하위 패턴:
  1. 소보행 (Shuffling gait) — 짧은 보폭 + 빠른 보행률
  2. 동결 보행 (Freezing of Gait, FOG) — 보폭 규칙성 급격 저하
  3. 전진 가속 (Festination) — 보행률 과다 + 속도 저하
  4. 자세 불안정 (Postural instability) — COP/체간 흔들림
  5. 운동완서 (Bradykinesia) — 전반적 운동 둔화

의학적 근거:
  - Hoehn & Yahr (1967) 파킨슨병 진행 단계
  - UPDRS Part III 운동 증상 평가 기반
  - Hausdorff et al. (2001) 보행 변동성 분석
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .common import compute_derived_features, linear_risk_score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파킨슨 하위 패턴 정의
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PARKINSONS_SUB_PATTERNS = {
    "shuffling": {
        "korean_name": "소보행 (Shuffling)",
        "description": "보폭이 짧고 발을 끌듯이 걷는 파킨슨 특징적 보행",
        "indicators": {
            "gait_speed": {"direction": "low", "normal": 1.0, "severe": 0.5, "weight": 1.0},
            "stride_regularity": {"direction": "low", "normal": 0.7, "severe": 0.35, "weight": 0.9},
            "acceleration_rms": {"direction": "low", "normal": 0.8, "severe": 0.3, "weight": 0.7},
            "heel_pressure_ratio": {"direction": "low", "normal": 0.25, "severe": 0.10, "weight": 0.6},
        },
        "clinical_meaning": "도파민 부족 → 기저핵 출력 억제 → 보폭 축소",
    },
    "freezing": {
        "korean_name": "동결 보행 (Freezing of Gait)",
        "description": "보행 중 갑작스러운 멈춤 또는 발이 바닥에 붙은 듯한 현상",
        "indicators": {
            "stride_regularity": {"direction": "low", "normal": 0.7, "severe": 0.30, "weight": 1.0},
            "acceleration_variability": {"direction": "high", "normal": 0.35, "severe": 0.70, "weight": 0.9},
            "step_symmetry": {"direction": "low", "normal": 0.85, "severe": 0.55, "weight": 0.7},
        },
        "clinical_meaning": "보행 자동화 회로 단절 → 보행 개시/유지 장애",
    },
    "festination": {
        "korean_name": "전진 가속 (Festination)",
        "description": "점차 빨라지는 짧은 걸음으로 제어 불능 상태에 빠지는 패턴",
        "indicators": {
            "cadence": {"direction": "high", "normal": 130, "severe": 170, "weight": 1.0},
            "gait_speed": {"direction": "low", "normal": 1.0, "severe": 0.5, "weight": 0.8},
            "stride_regularity": {"direction": "low", "normal": 0.7, "severe": 0.40, "weight": 0.7},
        },
        "clinical_meaning": "전경 자세 → 무게중심 전방 이동 → 보상적 가속",
    },
    "postural_instability": {
        "korean_name": "자세 불안정 (Postural Instability)",
        "description": "균형 유지 능력 저하로 낙상 위험이 증가하는 상태",
        "indicators": {
            "cop_sway": {"direction": "high", "normal": 0.06, "severe": 0.15, "weight": 1.0},
            "trunk_sway": {"direction": "high", "normal": 3.0, "severe": 6.0, "weight": 0.9},
            "ml_variability": {"direction": "high", "normal": 0.10, "severe": 0.22, "weight": 0.8},
            "step_symmetry": {"direction": "low", "normal": 0.85, "severe": 0.60, "weight": 0.6},
        },
        "clinical_meaning": "자세 반사 소실 → 외부 교란 시 균형 회복 불가",
    },
    "bradykinesia": {
        "korean_name": "운동완서 (Bradykinesia)",
        "description": "전반적인 운동 속도 저하 및 운동 진폭 감소",
        "indicators": {
            "gait_speed": {"direction": "low", "normal": 1.0, "severe": 0.4, "weight": 1.0},
            "acceleration_rms": {"direction": "low", "normal": 0.8, "severe": 0.3, "weight": 0.9},
            "cadence": {"direction": "low", "normal": 100, "severe": 65, "weight": 0.7},
        },
        "clinical_meaning": "운동 계획 및 실행 전반의 둔화 (cardinal sign)",
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hoehn & Yahr 단계 기준
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOEHN_YAHR_STAGES = [
    {
        "stage": 0,
        "label": "정상",
        "description": "파킨슨 증상 없음",
        "score_range": (0.0, 0.10),
    },
    {
        "stage": 1,
        "label": "1단계 — 일측성",
        "description": "한쪽에만 경미한 증상. 보행은 거의 정상이나 미세한 비대칭",
        "score_range": (0.10, 0.25),
    },
    {
        "stage": 2,
        "label": "2단계 — 양측성 (균형 유지)",
        "description": "양쪽 증상 발현. 보폭 감소, 보행률 변화 시작. 균형은 유지",
        "score_range": (0.25, 0.45),
    },
    {
        "stage": 3,
        "label": "3단계 — 자세 불안정 시작",
        "description": "자세 반사 장애 시작. 낙상 위험 증가. 일상생활은 독립 수행 가능",
        "score_range": (0.45, 0.65),
    },
    {
        "stage": 4,
        "label": "4단계 — 심각한 장애",
        "description": "독립 보행 가능하나 심하게 제한됨. 일상생활에 도움 필요",
        "score_range": (0.65, 0.85),
    },
    {
        "stage": 5,
        "label": "5단계 — 휠체어/와상",
        "description": "보조 없이는 보행 불가. 휠체어 또는 침대 의존",
        "score_range": (0.85, 1.01),
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 결과 데이터 구조
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SubPatternResult:
    """하위 보행 패턴 분석 결과."""
    pattern_id: str
    korean_name: str
    score: float                    # 0.0 ~ 1.0
    detected: bool                  # score > 임계값
    description: str
    clinical_meaning: str
    indicator_details: list[dict]   # 개별 지표 상세


@dataclass
class ParkinsonsReport:
    """파킨슨병 종합 분석 보고서."""
    risk_score: float               # 종합 위험도 0.0 ~ 1.0
    risk_label: str                 # 정상/경미/주의/경고/위험
    hoehn_yahr_stage: int           # 0~5
    hoehn_yahr_label: str           # 단계 설명
    hoehn_yahr_description: str
    sub_patterns: list[SubPatternResult]
    detected_patterns: list[SubPatternResult]  # detected=True 인 항목만
    key_findings: list[str]         # 주요 발견 사항 (한국어)
    recommendations: list[str]      # 진료/관리 권고
    confidence: float               # 판정 신뢰도 0.0 ~ 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 분석 엔진
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_DETECTION_THRESHOLD = 0.15


class ParkinsonsAnalyzer:
    """파킨슨병 특화 보행 분석 엔진.

    기존 범용 질환 예측(DiseaseRiskPredictor)과 달리,
    파킨슨 5대 하위 패턴을 개별 감지하고 Hoehn & Yahr 단계를 추정합니다.
    """

    def analyze(self, features: dict[str, float]) -> ParkinsonsReport:
        """보행 특성으로 파킨슨병 종합 분석을 수행합니다.

        Args:
            features: PersonalGaitProfiler.extract_session_features() 출력
                      또는 GaitFeatures dict.
        """
        features = dict(features)
        compute_derived_features(features)

        # 1. 하위 패턴별 평가
        sub_results = []
        available_total = 0
        used_total = 0

        for pattern_id, pattern_def in PARKINSONS_SUB_PATTERNS.items():
            result, available, used = self._evaluate_sub_pattern(
                pattern_id, pattern_def, features,
            )
            sub_results.append(result)
            available_total += available
            used_total += used

        # 2. 종합 위험도: 가중 평균 (detected 패턴에 높은 가중치)
        if sub_results:
            weights = []
            scores = []
            for r in sub_results:
                w = 2.0 if r.detected else 1.0
                weights.append(w)
                scores.append(r.score)
            risk_score = float(np.clip(
                np.average(scores, weights=weights), 0, 1
            ))
        else:
            risk_score = 0.0

        # 3. Hoehn & Yahr 단계 매핑
        hy_stage, hy_label, hy_desc = self._estimate_hoehn_yahr(risk_score, sub_results)

        # 4. 위험도 라벨
        risk_label = self._risk_label(risk_score)

        # 5. 감지된 패턴만 필터
        detected = [r for r in sub_results if r.detected]

        # 6. 주요 발견 사항
        findings = self._build_findings(detected, features, risk_score)

        # 7. 권고 사항
        recommendations = self._build_recommendations(
            risk_score, hy_stage, detected,
        )

        # 8. 신뢰도
        confidence = used_total / max(available_total, 1)

        return ParkinsonsReport(
            risk_score=round(risk_score, 4),
            risk_label=risk_label,
            hoehn_yahr_stage=hy_stage,
            hoehn_yahr_label=hy_label,
            hoehn_yahr_description=hy_desc,
            sub_patterns=sub_results,
            detected_patterns=detected,
            key_findings=findings,
            recommendations=recommendations,
            confidence=round(confidence, 4),
        )

    # ── 하위 패턴 평가 ──────────────────────────────────────────────

    def _evaluate_sub_pattern(
        self,
        pattern_id: str,
        pattern_def: dict,
        features: dict[str, float],
    ) -> tuple[SubPatternResult, int, int]:
        """단일 하위 패턴을 평가합니다.

        Returns:
            (SubPatternResult, total_indicators, available_indicators)
        """
        indicators = pattern_def["indicators"]
        total = len(indicators)
        used = 0
        scores = []
        weights = []
        details = []

        for feat_key, spec in indicators.items():
            if feat_key not in features:
                details.append({
                    "indicator": feat_key,
                    "status": "missing",
                    "score": 0.0,
                })
                continue

            used += 1
            value = features[feat_key]
            w = spec["weight"]
            normal_val = spec["normal"]
            severe_val = spec["severe"]

            if spec["direction"] == "low":
                score = linear_risk_score(value, normal_val, severe_val)
            else:
                score = linear_risk_score(value, normal_val, severe_val)

            scores.append(score)
            weights.append(w)
            details.append({
                "indicator": feat_key,
                "value": round(value, 4),
                "normal_threshold": normal_val,
                "severe_threshold": severe_val,
                "direction": spec["direction"],
                "score": round(score, 4),
                "status": "abnormal" if score > 0 else "normal",
            })

        if scores:
            pattern_score = float(np.clip(np.average(scores, weights=weights), 0, 1))
        else:
            pattern_score = 0.0

        return SubPatternResult(
            pattern_id=pattern_id,
            korean_name=pattern_def["korean_name"],
            score=round(pattern_score, 4),
            detected=pattern_score >= _DETECTION_THRESHOLD,
            description=pattern_def["description"],
            clinical_meaning=pattern_def["clinical_meaning"],
            indicator_details=details,
        ), total, used

    # ── Hoehn & Yahr 추정 ──────────────────────────────────────────

    def _estimate_hoehn_yahr(
        self,
        risk_score: float,
        sub_results: list[SubPatternResult],
    ) -> tuple[int, str, str]:
        """종합 위험도와 패턴 분포로 H&Y 단계를 추정합니다."""
        # 자세 불안정 감지 여부가 3단계 판별의 핵심 기준
        postural = next(
            (r for r in sub_results if r.pattern_id == "postural_instability"), None
        )
        postural_detected = postural is not None and postural.detected

        for stage_def in HOEHN_YAHR_STAGES:
            lo, hi = stage_def["score_range"]
            if lo <= risk_score < hi:
                stage = stage_def["stage"]
                # 3단계 보정: 자세 불안정이 감지되면 최소 3단계
                if postural_detected and stage < 3:
                    stage = 3
                    return 3, HOEHN_YAHR_STAGES[3]["label"], HOEHN_YAHR_STAGES[3]["description"]
                return stage, stage_def["label"], stage_def["description"]

        # fallback
        last = HOEHN_YAHR_STAGES[-1]
        return last["stage"], last["label"], last["description"]

    # ── 라벨링 ─────────────────────────────────────────────────────

    @staticmethod
    def _risk_label(score: float) -> str:
        if score >= 0.65:
            return "위험"
        elif score >= 0.45:
            return "경고"
        elif score >= 0.25:
            return "주의"
        elif score >= 0.10:
            return "경미"
        else:
            return "정상"

    # ── 발견 사항 생성 ─────────────────────────────────────────────

    def _build_findings(
        self,
        detected: list[SubPatternResult],
        features: dict[str, float],
        risk_score: float,
    ) -> list[str]:
        findings = []

        if not detected:
            findings.append("파킨슨 관련 보행 이상 패턴이 감지되지 않았습니다.")
            return findings

        findings.append(
            f"파킨슨 관련 보행 이상 {len(detected)}개 패턴 감지 "
            f"(종합 위험도 {risk_score:.0%})"
        )

        for r in sorted(detected, key=lambda x: -x.score):
            findings.append(f"  - {r.korean_name}: 심각도 {r.score:.0%}")

        gait_speed = features.get("gait_speed", None)
        if gait_speed is not None and gait_speed < 0.8:
            findings.append(
                f"보행 속도 {gait_speed:.2f} m/s — 정상 범위(1.0~1.4) 이하 "
                "(운동완서 가능성)"
            )

        cadence = features.get("cadence", None)
        if cadence is not None and cadence > 140:
            findings.append(
                f"보행률 {cadence:.0f} steps/min — 정상 범위(100~130) 초과 "
                "(소보행/가속보행 가능성)"
            )

        stride_reg = features.get("stride_regularity", None)
        if stride_reg is not None and stride_reg < 0.6:
            findings.append(
                f"보폭 규칙성 {stride_reg:.2f} — 정상 범위(0.7~1.0) 이하 "
                "(동결 보행 가능성)"
            )

        return findings

    # ── 권고 사항 생성 ─────────────────────────────────────────────

    def _build_recommendations(
        self,
        risk_score: float,
        hy_stage: int,
        detected: list[SubPatternResult],
    ) -> list[str]:
        recs = []

        if risk_score < 0.10:
            recs.append("현재 파킨슨 관련 보행 이상이 관찰되지 않습니다. 정기적 모니터링을 권장합니다.")
            return recs

        # 진료 권고
        if risk_score >= 0.45:
            recs.append("[필수] 신경과 전문의 진료를 받으십시오. 정밀 검사(DaTSCAN, MRI)를 권장합니다.")
        elif risk_score >= 0.25:
            recs.append("[권장] 신경과 전문의 상담을 고려하십시오.")
        else:
            recs.append("[참고] 경미한 이상 소견이 있습니다. 추적 관찰을 권장합니다.")

        # 패턴별 맞춤 권고
        detected_ids = {r.pattern_id for r in detected}

        if "shuffling" in detected_ids:
            recs.append(
                "[보행 훈련] 의식적으로 큰 보폭 유지 훈련 (Big Step Training). "
                "바닥에 일정 간격 표시 후 보폭 맞추기 연습."
            )

        if "freezing" in detected_ids:
            recs.append(
                "[동결 대처] 리듬 청각 자극(메트로놈, 음악) 활용 보행 훈련. "
                "레이저 포인터 시각 단서(visual cueing) 병행."
            )

        if "festination" in detected_ids:
            recs.append(
                "[가속 방지] 의식적 감속 훈련. 출발 전 '멈춤-생각-걸음' 3단계 연습. "
                "보행 보조기 사용 검토."
            )

        if "postural_instability" in detected_ids:
            recs.append(
                "[균형 훈련] 태극권/요가 등 균형 운동 프로그램 참여. "
                "낙상 예방 환경 정비(미끄럼 방지, 핸드레일). "
                "필요시 보행 보조기 처방."
            )

        if "bradykinesia" in detected_ids:
            recs.append(
                "[운동 프로그램] LSVT-BIG 운동 치료 (고진폭 운동 훈련). "
                "규칙적 유산소 운동(걷기, 자전거)으로 운동 기능 유지."
            )

        # H&Y 기반 추가 권고
        if hy_stage >= 3:
            recs.append(
                "[낙상 예방] Hoehn & Yahr 3단계 이상으로 낙상 위험 높음. "
                "보호자 동행 보행 및 주거 환경 안전 점검 필수."
            )

        if hy_stage >= 2:
            recs.append(
                "[약물 모니터링] 도파민 제제 효과 모니터링을 위해 "
                "투약 전후 보행 데이터를 정기적으로 비교 분석하십시오."
            )

        return recs
