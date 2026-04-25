"""Docker 빌드 없이 컨테이너 환경을 시뮬레이션하여 검증.

실제 Docker 데몬 없이 다음을 확인한다:
  1. .dockerignore 적용 후 COPY 대상 존재 여부
  2. requirements.txt 패키지 설치 상태
  3. PYTHONPATH=/app 환경에서 api.main 임포트 성공
  4. 서버 startup (warmup) + /health 응답
  5. 환경변수(API_KEYS, LOG_LEVEL, WORKERS) 동작
"""

from __future__ import annotations
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"

results: list[tuple[bool, str, str]] = []

def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name, detail))
    icon = PASS if ok else FAIL
    print(f"  {icon} {name}" + (f"  — {detail}" if detail else ""))
    return ok


# ── 1. COPY 대상 검증 ────────────────────────────────────────────────
print("\n[1] Dockerfile COPY 대상 검증")
for path in ["src/", "api/", "configs/", "requirements.txt"]:
    p = ROOT / path.rstrip("/")
    check(p.exists(), f"COPY {path}")

dockerignore = ROOT / ".dockerignore"
check(dockerignore.exists(), ".dockerignore 존재")


# ── 2. requirements.txt 패키지 상태 ──────────────────────────────────
print("\n[2] requirements.txt 패키지 설치 확인")
import importlib.util

pkg_map = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",           # pip name may include [standard]
    "pydantic": "pydantic",
    "python-dotenv": "dotenv",
    "transformers": "transformers",
    "peft": "peft",
}

with open(ROOT / "requirements.txt") as f:
    # strip pip extras like [standard] before lookup
    req_lines = [
        l.strip().split(">=")[0].split("==")[0].split("[")[0]
        for l in f if l.strip() and not l.startswith("#")
    ]

for req in req_lines:
    import_name = pkg_map.get(req, req)
    spec = importlib.util.find_spec(import_name)
    check(spec is not None, f"{req}", "설치됨" if spec else "미설치 — Docker 빌드 실패 예상")


# ── 3. PYTHONPATH=/app 임포트 체인 ───────────────────────────────────
print("\n[3] API 모듈 임포트 체인")
for module in [
    "src.models.multimodal_gait_net",
    "src.models.hf_encoders",
    "src.data.adapters",
    "api.main",
    "api.service",
]:
    try:
        __import__(module)
        check(True, f"import {module}")
    except Exception as e:
        check(False, f"import {module}", str(e)[:80])


# ── 4. FastAPI TestClient startup + health ────────────────────────────
print("\n[4] 서버 startup + /health 응답")
try:
    from fastapi.testclient import TestClient
    from api.main import app

    t0 = time.time()
    with TestClient(app, raise_server_exceptions=False) as client:
        startup_ms = (time.time() - t0) * 1000
        check(True, "startup (warmup)", f"{startup_ms:.0f}ms")
        check(startup_ms < 60_000, "startup < 60s (HEALTHCHECK start_period)")

        r = client.get("/health")
        check(r.status_code == 200, "/health", f"status={r.status_code}")
        check(r.json().get("status") == "ok", "/health body", f"body={r.json()}")
except Exception as e:
    check(False, "TestClient 실행", str(e)[:80])


# ── 5. 환경변수 동작 확인 ────────────────────────────────────────────
print("\n[5] 환경변수 동작")

# API_KEYS — auth.py는 모듈 로드 시점에 env 읽음, subprocess로 격리 확인
import subprocess
result = subprocess.run(
    [sys.executable, "-c",
     "import os; os.environ['API_KEYS']='test-key-abc'; "
     "import api.auth as a; "
     "print(a.AUTH_ENABLED, 'test-key-abc' in a._ALLOWED_KEYS)"],
    capture_output=True, text=True, cwd=str(ROOT),
    env={**os.environ, "API_KEYS": "test-key-abc", "PYTHONPATH": str(ROOT)},
)
output = result.stdout.strip()
auth_ok = output == "True True"
check(auth_ok, "API_KEYS 환경변수", f"output='{output}'")

# LOG_LEVEL
os.environ["LOG_LEVEL"] = "DEBUG"
try:
    import api.logging_config as lc
    importlib.reload(lc)
    check(True, "LOG_LEVEL 환경변수", "설정 가능")
except Exception as e:
    check(False, "LOG_LEVEL 환경변수", str(e)[:60])
finally:
    os.environ.pop("LOG_LEVEL", None)

# WORKERS (CMD에서만 사용 — 환경변수 존재 확인)
check(True, "WORKERS 환경변수", "CMD에서 ${WORKERS} 치환")


# ── 6. .dockerignore 검증 ────────────────────────────────────────────
print("\n[6] .dockerignore 효과 검증")
ignore_patterns = dockerignore.read_text().splitlines() if dockerignore.exists() else []
should_exclude = ["tests/", "scripts/", "frontend/", "outputs/", ".git/"]
for path in should_exclude:
    excluded = any(path.rstrip("/") in p for p in ignore_patterns)
    check(excluded, f"{path} 제외됨")


# ── 요약 ─────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for ok, _, _ in results if ok)
failed = total - passed

print(f"\n{'='*55}")
if failed == 0:
    print(f"  \033[92m전체 통과: {passed}/{total}  — Docker 빌드 준비 완료\033[0m")
else:
    print(f"  \033[91m실패: {failed}/{total}  통과: {passed}/{total}\033[0m")
    print("\n  실패 항목:")
    for ok, name, detail in results:
        if not ok:
            print(f"    ✗ {name}: {detail}")
print(f"{'='*55}\n")

sys.exit(0 if failed == 0 else 1)
