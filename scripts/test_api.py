"""API 배포 테스트 스크립트.

FastAPI TestClient로 모든 엔드포인트를 순차 검증한다:
  /health
  /api/v1/sample
  /api/v1/classify
  /api/v1/disease-risk
  /api/v1/injury-risk
  /api/v1/reasoning
  /api/v1/analyze
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m·\033[0m"


def _check(name: str, condition: bool, detail: str = "") -> bool:
    icon = PASS if condition else FAIL
    extra = f"  {detail}" if detail else ""
    print(f"  {icon} {name}{extra}")
    return condition


def run_tests():
    print("\n" + "=" * 60)
    print("  Shoealls API 배포 테스트")
    print("=" * 60)

    passed = 0
    failed = 0

    with TestClient(app, raise_server_exceptions=False) as client:

        # ── 1. Health ───────────────────────────────────────────────
        print("\n[1] Health Check")
        t = time.time()
        r = client.get("/health")
        elapsed = time.time() - t
        ok = r.status_code == 200 and r.json().get("status") == "ok"
        if _check("/health", ok, f"status={r.status_code} ({elapsed*1000:.0f}ms)"):
            passed += 1
        else:
            failed += 1
            print(f"     응답: {r.text[:200]}")

        # ── 2. Sample Data ──────────────────────────────────────────
        print("\n[2] Sample Data 생성")
        samples = {}
        for profile in ["normal", "parkinsons", "stroke", "fall_risk"]:
            r = client.get(f"/api/v1/sample?gait_profile={profile}")
            ok = r.status_code == 200 and "sensor_data" in r.json()
            if _check(f"/api/v1/sample?profile={profile}", ok, f"status={r.status_code}"):
                passed += 1
                samples[profile] = r.json()
            else:
                failed += 1
                print(f"     응답: {r.text[:200]}")

        if not samples:
            print("\n  샘플 데이터 없음 — 이후 테스트 건너뜀")
            _print_summary(passed, failed)
            return

        normal_sample = samples.get("normal", list(samples.values())[0])
        sensor_data = normal_sample["sensor_data"]
        features = normal_sample["features"]

        # ── 3. Classify ────────────────────────────────────────────
        print("\n[3] 보행 분류 (/api/v1/classify)")
        t = time.time()
        r = client.post("/api/v1/classify", json={"sensor_data": sensor_data})
        elapsed = time.time() - t
        if r.status_code == 200:
            data = r.json()
            ok = "prediction" in data and "confidence" in data
            _check("classify", ok, f"pred={data.get('prediction')} conf={data.get('confidence', 0):.3f} ({elapsed*1000:.0f}ms)")
            if ok:
                passed += 1
                print(f"     {INFO} class_probs: { {k: round(v,3) for k,v in data['class_probabilities'].items()} }")
                print(f"     {INFO} demo_mode: {data.get('is_demo_mode')}")
            else:
                failed += 1
        else:
            _check("classify", False, f"status={r.status_code}")
            failed += 1
            print(f"     응답: {r.text[:300]}")

        # ── 4. Disease Risk ────────────────────────────────────────
        # parkinsons 프로파일로 테스트 (질환이 더 잘 나옴)
        print("\n[4] 질환 위험도 (/api/v1/disease-risk)")
        pk_features = samples.get("parkinsons", normal_sample)["features"]
        t = time.time()
        r = client.post("/api/v1/disease-risk", json={"features": pk_features})
        elapsed = time.time() - t
        if r.status_code == 200:
            data = r.json()
            ok = "top_diseases" in data and "ml_prediction" in data
            top = data["top_diseases"][0] if data["top_diseases"] else {}
            _check("disease-risk", ok,
                   f"top={top.get('disease_kr','(정상)')} ({top.get('risk_score',0):.3f}) ({elapsed*1000:.0f}ms)")
            if ok:
                passed += 1
                print(f"     {INFO} ml_pred: {data.get('ml_prediction_kr')} conf={data.get('ml_confidence',0):.3f}")
                print(f"     {INFO} abnormal: {data.get('abnormal_biomarkers', [])[:3]}")
            else:
                failed += 1
        else:
            _check("disease-risk", False, f"status={r.status_code}")
            failed += 1
            print(f"     응답: {r.text[:300]}")

        # ── 5. Injury Risk ─────────────────────────────────────────
        print("\n[5] 부상 위험 (/api/v1/injury-risk)")
        t = time.time()
        r = client.post("/api/v1/injury-risk", json={"features": features})
        elapsed = time.time() - t
        if r.status_code == 200:
            data = r.json()
            ok = "predicted_injury" in data and "combined_risk_score" in data
            _check("injury-risk", ok,
                   f"injury={data.get('predicted_injury_kr','?')} "
                   f"risk={data.get('combined_risk_score',0):.3f} "
                   f"grade={data.get('combined_risk_grade','?')} "
                   f"({elapsed*1000:.0f}ms)")
            if ok:
                passed += 1
                print(f"     {INFO} top3: {[(x['name_kr'], round(x['probability'],3)) for x in data.get('top3',[])[:2]]}")
            else:
                failed += 1
        else:
            _check("injury-risk", False, f"status={r.status_code}")
            failed += 1
            print(f"     응답: {r.text[:300]}")

        # ── 6. Reasoning ───────────────────────────────────────────
        print("\n[6] Chain-of-Reasoning (/api/v1/reasoning)")
        t = time.time()
        r = client.post("/api/v1/reasoning", json={"sensor_data": sensor_data})
        elapsed = time.time() - t
        if r.status_code == 200:
            data = r.json()
            ok = "final_prediction" in data and "reasoning_trace" in data
            _check("reasoning", ok,
                   f"pred={data.get('final_prediction')} "
                   f"conf={data.get('confidence',0):.3f} "
                   f"uncertainty={data.get('uncertainty',0):.3f} "
                   f"({elapsed*1000:.0f}ms)")
            if ok:
                passed += 1
                trace = data.get("reasoning_trace", [])
                print(f"     {INFO} trace steps: {len(trace)}")
                print(f"     {INFO} evidence_strength: {data.get('evidence_strength',0):.3f}")
            else:
                failed += 1
        else:
            _check("reasoning", False, f"status={r.status_code}")
            failed += 1
            print(f"     응답: {r.text[:300]}")

        # ── 7. Full Analyze ────────────────────────────────────────
        print("\n[7] 종합 분석 (/api/v1/analyze)")
        t = time.time()
        r = client.post("/api/v1/analyze", json={
            "sensor_data": sensor_data,
            "features": features,
        })
        elapsed = time.time() - t
        if r.status_code == 200:
            data = r.json()
            ok = all(k in data for k in ["classify", "disease_risk", "injury_risk", "reasoning"])
            _check("analyze", ok, f"({elapsed*1000:.0f}ms) keys={list(data.keys())}")
            if ok:
                passed += 1
            else:
                failed += 1
        else:
            _check("analyze", False, f"status={r.status_code}")
            failed += 1
            print(f"     응답: {r.text[:300]}")

        # ── 8. 에러 케이스 ─────────────────────────────────────────
        print("\n[8] 에러 핸들링")
        r = client.get("/api/v1/sample?gait_profile=invalid_profile")
        ok = r.status_code == 400
        if _check("잘못된 profile → 400", ok, f"status={r.status_code}"):
            passed += 1
        else:
            failed += 1

        r = client.post("/api/v1/classify", json={})
        ok = r.status_code == 422  # Pydantic validation
        if _check("빈 body → 422", ok, f"status={r.status_code}"):
            passed += 1
        else:
            failed += 1

    _print_summary(passed, failed)
    return failed == 0


def _print_summary(passed: int, failed: int):
    total = passed + failed
    print("\n" + "=" * 60)
    if failed == 0:
        print(f"  \033[92m전체 통과: {passed}/{total}\033[0m")
    else:
        print(f"  \033[91m실패: {failed}/{total}  통과: {passed}/{total}\033[0m")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
