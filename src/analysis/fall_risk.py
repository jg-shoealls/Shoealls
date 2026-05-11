"""Fall Risk Assessment Engine.

Evaluates the probability of falls based on dynamic stability, 
symmetry, and rhythm biomarkers.
"""

import numpy as np
from dataclasses import dataclass
from .common import linear_risk_score, severity_label_4level

@dataclass
class FallRiskReport:
    risk_score: float        # 0.0 ~ 1.0
    risk_level: str          # 정상, 주의, 경고, 위험
    stability_score: float   # 0 ~ 100
    symmetry_score: float    # 0 ~ 100
    rhythm_score: float      # 0 ~ 100
    risk_factors: list[str]
    recommendations: list[str]

class FallRiskEngine:
    """Calculates fall risk using spatio-temporal and pressure biomarkers."""

    def __init__(self):
        pass

    def assess(self, features: dict[str, float]) -> FallRiskReport:
        """
        Assess fall risk from extracted gait features.
        
        Features expected:
            - gait_speed: (m/s)
            - cop_sway: (normalized)
            - ml_variability: (std)
            - step_symmetry: (ratio)
            - stride_regularity: (ratio)
            - cadence: (steps/min)
        """
        factors = []
        scores = []

        # 1. Stability (COP Sway & ML Variability)
        cop_sway = features.get("cop_sway", 0.04)
        s_cop = linear_risk_score(cop_sway, 0.06, 0.12)
        if s_cop > 0.3:
            factors.append("체중심 흔들림(COP Sway) 과다")
        
        ml_var = features.get("ml_variability", 0.05)
        s_ml = linear_risk_score(ml_var, 0.10, 0.20)
        if s_ml > 0.3:
            factors.append("좌우 보행 변동성 높음")
        
        stability_risk = (s_cop + s_ml) / 2

        # 2. Symmetry (Step Symmetry)
        symmetry = features.get("step_symmetry", 0.9)
        s_sym = linear_risk_score(symmetry, 0.85, 0.65) # Lower is riskier
        if s_sym > 0.3:
            factors.append("좌우 보행 비대칭성 감지")
        
        symmetry_risk = s_sym

        # 3. Rhythm (Stride Regularity & Cadence)
        regularity = features.get("stride_regularity", 0.8)
        s_reg = linear_risk_score(regularity, 0.70, 0.50)
        if s_reg > 0.3:
            factors.append("보행 규칙성(Regularity) 저하")
            
        cadence = features.get("cadence", 110)
        # Very low or very high cadence can be risky
        if cadence < 90:
            s_cad = linear_risk_score(cadence, 90, 70)
            if s_cad > 0.3: factors.append("느린 보행 속도(서행)")
        elif cadence > 140:
            s_cad = linear_risk_score(cadence, 140, 160)
            if s_cad > 0.3: factors.append("비정상적으로 빠른 보행(충동 보행)")
        else:
            s_cad = 0.0
            
        rhythm_risk = (s_reg + s_cad) / 2

        # 4. Gait Speed (Crucial for fall risk)
        speed = features.get("gait_speed", 1.2)
        s_speed = linear_risk_score(speed, 1.0, 0.6)
        if s_speed > 0.3:
            factors.append("보행 속도 저하 (낙상 고위험군 지표)")

        # Combined Risk Score (Weighted)
        # Gait speed and Stability are usually the strongest predictors
        total_risk = (
            stability_risk * 0.35 +
            s_speed * 0.30 +
            symmetry_risk * 0.15 +
            rhythm_risk * 0.20
        )
        total_risk = float(np.clip(total_risk, 0, 1))

        # Recommendations
        recs = []
        if total_risk > 0.2:
            recs.append("하지 근력 강화 운동(스쿼트, 까치발 들기) 권장")
        if total_risk > 0.4:
            recs.append("균형 감각 개선 훈련(한 발 서기) 권장")
            recs.append("실내 낙상 방지를 위한 미끄럼 방지 매트 사용 권장")
        if total_risk > 0.6:
            recs.append("보행 보조 기구(지팡이 등) 사용 검토 권장")
            recs.append("전문 의료기관 방문을 통한 정밀 균형 검사 권고")
        
        if not recs:
            recs.append("현재의 규칙적인 보행 습관 유지")

        return FallRiskReport(
            risk_score=total_risk,
            risk_level=severity_label_4level(total_risk),
            stability_score=round((1 - stability_risk) * 100, 1),
            symmetry_score=round((1 - symmetry_risk) * 100, 1),
            rhythm_score=round((1 - rhythm_risk) * 100, 1),
            risk_factors=factors if factors else ["특이 소견 없음"],
            recommendations=recs
        )
