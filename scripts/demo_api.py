"""API 동작 검증 스크립트 (서버 없이 서비스 레이어 직접 테스트).

Usage:
    python scripts/demo_api.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from api.schemas import SensorData, GaitFeatures
from api.service import GaitMLService


def make_sensor_data(gait_class: int = 0) -> SensorData:
    """합성 센서 데이터 생성."""
    rng = np.random.default_rng(42 + gait_class)
    seq_len = 128

    # IMU: [128, 6]
    freq = {0: 1.8, 1: 1.2, 2: 1.5, 3: 1.0}[gait_class]
    t = np.linspace(0, seq_len / 30.0, seq_len)
    imu = np.stack([
        np.sin(2 * np.pi * freq * t + i * np.pi / 6) + 0.1 * rng.standard_normal(seq_len)
        for i in range(6)
    ], axis=1).tolist()

    # Pressure: [16, 8]
    pressure = (rng.random((16, 8)) * 0.5 + 0.1).tolist()

    # Skeleton: [128, 17, 3]
    skeleton = (rng.random((seq_len, 17, 3))).tolist()

    return SensorData(imu=imu, pressure=pressure, skeleton=skeleton)


NORMAL_FEATURES = GaitFeatures(
    gait_speed=1.25, cadence=118, stride_regularity=0.88,
    step_symmetry=0.93, cop_sway=0.035, ml_index=0.03,
    arch_index=0.24, acceleration_rms=1.6,
    zone_heel_medial_mean=0.32, zone_heel_lateral_mean=0.30,
    zone_forefoot_medial_mean=0.33, zone_forefoot_lateral_mean=0.30,
    zone_toes_mean=0.18, zone_midfoot_medial_mean=0.10,
    zone_midfoot_lateral_mean=0.08,
)

PARKINSONS_FEATURES = GaitFeatures(
    gait_speed=0.65, cadence=148, stride_regularity=0.42,
    step_symmetry=0.78, cop_sway=0.065, ml_index=0.08,
    arch_index=0.23, acceleration_rms=0.9,
    zone_heel_medial_mean=0.28, zone_heel_lateral_mean=0.25,
    zone_forefoot_medial_mean=0.38, zone_forefoot_lateral_mean=0.35,
    zone_toes_mean=0.12, zone_midfoot_medial_mean=0.10,
    zone_midfoot_lateral_mean=0.10,
)


def main():
    print("=" * 70)
    print("  Shoealls Gait Analysis API — 서비스 레이어 직접 테스트")
    print("=" * 70)

    svc = GaitMLService()
    print("\n[초기화] sklearn 모델 학습 중...")
    svc.warmup()
    print("  완료!")

    sensor = make_sensor_data(gait_class=0)

    # ── 1. 보행 분류 ──────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  [1] 보행 패턴 분류")
    print("─" * 50)
    result = svc.classify(sensor)
    print(f"  예측: {result.prediction_kr} (신뢰도 {result.confidence:.1%})")
    print(f"  데모 모드: {result.is_demo_mode}")
    for cls, prob in result.class_probabilities.items():
        bar = "█" * int(prob * 20)
        print(f"    {cls:12s} [{bar:<20s}] {prob:.1%}")

    # ── 2. 질환 위험도 (정상) ─────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  [2] 질환 위험도 예측 — 정상 보행")
    print("─" * 50)
    result = svc.disease_risk(NORMAL_FEATURES)
    print(f"  ML 예측: {result.ml_prediction_kr} (신뢰도 {result.ml_confidence:.1%})")
    if result.top_diseases:
        print(f"  상위 위험 질환: {result.top_diseases[0].disease_kr} ({result.top_diseases[0].risk_score:.1%})")
    else:
        print("  주요 위험 질환 없음 (정상)")
    print(f"  이상 바이오마커: {result.abnormal_biomarkers or '없음'}")

    # ── 3. 질환 위험도 (파킨슨 의심) ──────────────────────────────────
    print("\n" + "─" * 50)
    print("  [3] 질환 위험도 예측 — 파킨슨 의심")
    print("─" * 50)
    result = svc.disease_risk(PARKINSONS_FEATURES)
    print(f"  ML 예측: {result.ml_prediction_kr} (신뢰도 {result.ml_confidence:.1%})")
    print("  Top 3 질환:")
    for item in result.ml_top3[:3]:
        print(f"    {item['name_kr']:15s} {item['probability']:.1%}")
    print(f"  이상 바이오마커: {result.abnormal_biomarkers}")

    # ── 4. 부상 위험 ─────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  [4] 부상 위험 예측")
    print("─" * 50)
    result = svc.injury_risk(NORMAL_FEATURES)
    print(f"  예측 부상: {result.predicted_injury_kr} (신뢰도 {result.confidence:.1%})")
    print(f"  종합 위험도: {result.combined_risk_score:.1%} [{result.combined_risk_grade}]")
    print(f"  부상 시기: {result.timeline}")
    if result.priority_actions:
        print(f"  우선 조치: {result.priority_actions[0]}")

    # ── 5. Chain-of-Reasoning ─────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  [5] Chain-of-Reasoning 추론 분석")
    print("─" * 50)
    result = svc.reasoning(sensor)
    print(f"  최종 판정: {result.final_prediction_kr} (확신도 {result.confidence:.1%})")
    print(f"  불확실성: {result.uncertainty:.1%}")
    print(f"  근거 강도: {result.evidence_strength:.1%}")
    print("  추론 단계:")
    for step in result.reasoning_trace:
        print(f"    {step.label}: {step.prediction_kr} ({step.probability:.1%})")

    print("\n" + "=" * 70)
    print("  서버 실행: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
    print("  API 문서: http://localhost:8000/docs")
    print("=" * 70)


if __name__ == "__main__":
    main()
