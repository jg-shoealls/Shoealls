"""실제 학습된 모델로 예측 — demo 모드 vs 체크포인트 모드 비교.

사용법:
    python scripts/run_real_predict.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app

CKPT = str(Path(__file__).parent.parent / "outputs" / "best_model.pt")

PROFILES = ["normal", "parkinsons", "stroke", "fall_risk"]
PROFILE_LABEL = {
    "normal":    "정상 보행",
    "parkinsons":"파킨슨 보행",
    "stroke":    "뇌졸중 보행",
    "fall_risk": "낙상 위험 보행",
}
CLASS_KR = {
    "normal":       "정상",
    "antalgic":     "절뚝거림",
    "ataxic":       "운동실조",
    "parkinsonian": "파킨슨",
}
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def run():
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  Shoealls — 실제 모델 예측 (체크포인트: best_model.pt){RESET}")
    print(f"{BOLD}{'='*65}{RESET}\n")

    ckpt_exists = Path(CKPT).exists()
    print(f"  체크포인트: {'✓ 있음' if ckpt_exists else '✗ 없음 (demo 모드)'}")
    if not ckpt_exists:
        print(f"  경로: {CKPT}")
        return

    with TestClient(app, raise_server_exceptions=True) as client:
        # 샘플 데이터 수집
        samples = {}
        for p in PROFILES:
            r = client.get(f"/api/v1/sample?gait_profile={p}")
            samples[p] = r.json()

        # ── 분류 비교: demo vs 체크포인트 ──────────────────────────────
        print(f"\n{BOLD}[1] 보행 분류 — Demo(랜덤) vs 학습된 모델{RESET}")
        print(f"  {'프로파일':<14} {'Demo 예측':^18} {'학습 모델 예측':^20} {'신뢰도':>6}")
        print(f"  {'─'*14} {'─'*18} {'─'*20} {'─'*6}")

        for profile in PROFILES:
            sd = samples[profile]["sensor_data"]
            label = PROFILE_LABEL[profile]

            # demo 모드
            r_demo = client.post("/api/v1/classify", json={"sensor_data": sd})
            d_demo = r_demo.json()
            demo_pred = CLASS_KR.get(d_demo["prediction"], d_demo["prediction"])
            demo_conf = d_demo["confidence"]

            # 체크포인트 모드
            r_real = client.post("/api/v1/classify", json={
                "sensor_data": sd,
                "checkpoint_path": CKPT,
            })
            d_real = r_real.json()
            real_pred = CLASS_KR.get(d_real["prediction"], d_real["prediction"])
            real_conf = d_real["confidence"]

            conf_color = GREEN if real_conf >= 0.6 else YELLOW
            print(f"  {label:<14} {demo_pred:^18} {real_pred:^20} {conf_color}{real_conf:.1%}{RESET}")

        # ── 상세 예측: 4개 프로파일 × 클래스별 확률 ────────────────────
        print(f"\n{BOLD}[2] 클래스별 확률 분포 (학습된 모델){RESET}")
        for profile in PROFILES:
            sd = samples[profile]["sensor_data"]
            r = client.post("/api/v1/classify", json={
                "sensor_data": sd,
                "checkpoint_path": CKPT,
            })
            d = r.json()
            label = PROFILE_LABEL[profile]
            pred  = CLASS_KR.get(d["prediction"], d["prediction"])
            conf  = d["confidence"]

            print(f"\n  {CYAN}{label}{RESET}  →  예측: {BOLD}{pred}{RESET}  ({conf:.1%})")
            for cls_en, prob in sorted(d["class_probabilities"].items(),
                                       key=lambda x: -x[1]):
                cls_kr = CLASS_KR.get(cls_en, cls_en)
                highlight = BOLD + GREEN if cls_en == d["prediction"] else ""
                print(f"    {highlight}{cls_kr:<8}{RESET}  {bar(prob)}  {prob:.1%}")

        # ── Chain-of-Reasoning (파킨슨 프로파일) ───────────────────────
        print(f"\n{BOLD}[3] Chain-of-Reasoning — 파킨슨 보행{RESET}")
        sd = samples["parkinsons"]["sensor_data"]
        r = client.post("/api/v1/reasoning", json={
            "sensor_data": sd,
            "checkpoint_path": CKPT,
        })
        d = r.json()
        pred_kr = CLASS_KR.get(d["final_prediction"], d["final_prediction"])
        print(f"  최종 판정: {BOLD}{pred_kr}{RESET}  "
              f"신뢰도={d['confidence']:.1%}  불확실성={d['uncertainty']:.3f}")
        print(f"  근거 강도: {d['evidence_strength']:.3f}")
        print(f"\n  추론 단계:")
        for step in d["reasoning_trace"]:
            step_pred = CLASS_KR.get(step["prediction"], step["prediction"])
            print(f"    {step['step']}단계 [{step['label']}]  {step_pred}  ({step['probability']:.1%})")

        print(f"\n  이상 감지:")
        for mod in d["anomaly_findings"]:
            anomalies = mod["anomalies"]
            if anomalies:
                names = [a["type"] for a in anomalies[:2]]
                print(f"    {mod['modality']}: {', '.join(names)}")
            else:
                print(f"    {mod['modality']}: 이상 없음")

        print(f"\n  모달리티 가중치: {d['modality_weights']}")

        # ── 종합 분석 (낙상 위험 프로파일) ─────────────────────────────
        print(f"\n{BOLD}[4] 종합 분석 — 낙상 위험 보행{RESET}")
        sd   = samples["fall_risk"]["sensor_data"]
        feat = samples["fall_risk"]["features"]
        r = client.post("/api/v1/analyze", json={
            "sensor_data": sd,
            "features":    feat,
            "checkpoint_path": CKPT,
        })
        d = r.json()

        clf = d["classify"]
        dis = d["disease_risk"]
        inj = d["injury_risk"]

        print(f"  보행 분류:  {CLASS_KR.get(clf['prediction'], clf['prediction'])}  ({clf['confidence']:.1%})")
        if dis["top_diseases"]:
            top_d = dis["top_diseases"][0]
            print(f"  주요 질환:  {top_d['disease_kr']}  ({top_d['risk_score']:.1%} 위험도, {top_d['severity']})")
        print(f"  ML 질환:   {dis['ml_prediction_kr']}  ({dis['ml_confidence']:.1%})")
        print(f"  부상 예측:  {inj['predicted_injury_kr']}  (종합 위험: {inj['combined_risk_score']:.1%} / {inj['combined_risk_grade']})")
        if inj.get("priority_actions"):
            print(f"  우선 조치:  {inj['priority_actions'][0]}")

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"  완료")
    print(f"{BOLD}{'='*65}{RESET}\n")


if __name__ == "__main__":
    run()
