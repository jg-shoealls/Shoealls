"""전처리 파이프라인 경계 검증 스크립트.

실제 센서 데이터 수신 전 반드시 통과해야 할 케이스:
  - 정상 데이터 (다양한 시퀀스 길이)
  - NaN / Inf (센서 연결 불안정, 측정 범위 초과)
  - 모든 값 동일 / 0 (센서 포화, 발을 뗀 상태)
  - 타입 이상 (int16, int32 등)
  - 타임스탬프 열 자동 제거 (IMU 7채널)
  - 2D 스켈레톤 (z=0 패딩)
  - 의도적 오류 — 잘못된 입력에 명확한 ValueError 반환 확인
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.preprocessing import preprocess_imu, preprocess_pressure, preprocess_skeleton

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
rng = np.random.default_rng(42)

results: list[tuple[bool, str, str]] = []


def expect_ok(name: str, fn, *args) -> bool:
    """정상 처리 기대: NaN/Inf 없는 올바른 shape 반환."""
    try:
        out = fn(*args)
        ok = np.isfinite(out).all()
        detail = f"shape={out.shape}" if ok else "NaN/Inf in output"
    except Exception as e:
        ok = False
        detail = f"{type(e).__name__}: {e}"
    results.append((ok, name, detail))
    print(f"  {PASS if ok else FAIL} {name}" + (f"  → {detail}" if not ok else ""))
    return ok


def expect_err(name: str, exc_type, fn, *args) -> bool:
    """의도적 오류 기대: 명확한 예외 발생 + 유용한 메시지."""
    try:
        fn(*args)
        ok = False
        detail = "예외 미발생 — 잘못된 입력이 통과됨"
    except exc_type as e:
        ok = True
        detail = str(e)[:80]
    except Exception as e:
        ok = False
        detail = f"잘못된 예외 {type(e).__name__}: {e}"
    results.append((ok, name, detail))
    icon = PASS if ok else FAIL
    print(f"  {icon} {name}" + (f"  → {detail}" if not ok else f"  → {type(exc_type).__name__ if isinstance(exc_type, type) else ''}"))
    return ok


# ─── IMU ─────────────────────────────────────────────────────────────────────
print("\n[IMU] preprocess_imu")

expect_ok("정상 (128,6)",                   preprocess_imu, rng.random((128,6)).astype(np.float32), 128)
expect_ok("업샘플 30Hz→30Hz  (32,6)→128",  preprocess_imu, rng.random((32,6)).astype(np.float32),  128)
expect_ok("다운샘플 (512,6)→128",           preprocess_imu, rng.random((512,6)).astype(np.float32), 128)
expect_ok("T=1 극단 케이스",                preprocess_imu, rng.random((1,6)).astype(np.float32),   128)

d = rng.random((128,6)).astype(np.float32); d[10,2] = np.nan; d[50,0] = np.nan
expect_ok("NaN 다수 포함",                  preprocess_imu, d, 128)

d = rng.random((128,6)).astype(np.float32); d[5,0] = np.inf; d[30,3] = -np.inf
expect_ok("±Inf 포함",                      preprocess_imu, d, 128)

expect_ok("모든 값 동일 (std=0)",           preprocess_imu, np.ones((128,6),np.float32)*3.0, 128)
expect_ok("int16 dtype",                    preprocess_imu, (rng.random((128,6))*1000).astype(np.int16), 128)
expect_ok("타임스탬프 포함 7채널 자동 제거", preprocess_imu, rng.random((128,7)).astype(np.float32), 128)

expect_err("채널 4개 → ValueError",         ValueError, preprocess_imu, rng.random((128,4)).astype(np.float32), 128)
expect_err("1-D 배열 → ValueError",         ValueError, preprocess_imu, rng.random(128).astype(np.float32), 128)


# ─── Pressure ────────────────────────────────────────────────────────────────
print("\n[Pressure] preprocess_pressure")

expect_ok("정상 3D (128,16,8)",             preprocess_pressure, rng.random((128,16,8)).astype(np.float32), 128, (16,8))
expect_ok("2D flat (128,128)",              preprocess_pressure, rng.random((128,128)).astype(np.float32),  128, (16,8))
expect_ok("음수 포함 (센서 오프셋)",         preprocess_pressure, rng.random((128,16,8)).astype(np.float32)-0.5, 128, (16,8))
expect_ok("모든 값 0 (발 뗀 상태)",         preprocess_pressure, np.zeros((128,16,8),np.float32), 128, (16,8))

d = rng.random((128,16,8)).astype(np.float32); d[0,0,0] = np.nan; d[10,8,4] = np.inf
expect_ok("NaN+Inf 혼합",                   preprocess_pressure, d, 128, (16,8))

expect_ok("업샘플 (32,16,8)→128",          preprocess_pressure, rng.random((32,16,8)).astype(np.float32), 128, (16,8))

expect_err("그리드 불일치 (128,32) → ValueError", ValueError,
           preprocess_pressure, rng.random((128,32)).astype(np.float32), 128, (16,8))


# ─── Skeleton ────────────────────────────────────────────────────────────────
print("\n[Skeleton] preprocess_skeleton")

expect_ok("정상 3D (128,17,3)",             preprocess_skeleton, rng.random((128,17,3)).astype(np.float32), 128, 17)

d = np.zeros((128,17,3),np.float32); d[:,1:,:] = rng.random((128,16,3)).astype(np.float32)
expect_ok("hip=0 (scale=0 안정성)",         preprocess_skeleton, d, 128, 17)

expect_ok("2D 스켈레톤 (128,17,2) z패딩",  preprocess_skeleton, rng.random((128,17,2)).astype(np.float32), 128, 17)

d = rng.random((128,17,3)).astype(np.float32); d[20,5,1] = np.nan; d[70,0,2] = np.inf
expect_ok("NaN+Inf 혼합",                   preprocess_skeleton, d, 128, 17)

expect_ok("T=2 극단 케이스",               preprocess_skeleton, rng.random((2,17,3)).astype(np.float32), 128, 17)
expect_ok("다운샘플 (500,17,3)→128",       preprocess_skeleton, rng.random((500,17,3)).astype(np.float32), 128, 17)

expect_err("관절 수 불일치 (128,33,3) → ValueError", ValueError,
           preprocess_skeleton, rng.random((128,33,3)).astype(np.float32), 128, 17)
expect_err("좌표 차원 4 → ValueError",     ValueError,
           preprocess_skeleton, rng.random((128,17,4)).astype(np.float32), 128, 17)


# ─── 출력 형태 검증 ─────────────────────────────────────────────────────────
print("\n[Output Shape] 모델 입력 형태 일치 확인")

imu_out  = preprocess_imu(rng.random((128,6)).astype(np.float32), 128)
pres_out = preprocess_pressure(rng.random((128,16,8)).astype(np.float32), 128, (16,8))
skel_out = preprocess_skeleton(rng.random((128,17,3)).astype(np.float32), 128, 17)

ok = imu_out.shape  == (6, 128);  results.append((ok, "IMU shape (6,128)", str(imu_out.shape)))
print(f"  {PASS if ok else FAIL} IMU  → {imu_out.shape}  (expect (6,128))")

ok = pres_out.shape == (128, 1, 16, 8); results.append((ok, "Pressure shape (128,1,16,8)", str(pres_out.shape)))
print(f"  {PASS if ok else FAIL} Pressure → {pres_out.shape}  (expect (128,1,16,8))")

ok = skel_out.shape == (3, 128, 17); results.append((ok, "Skeleton shape (3,128,17)", str(skel_out.shape)))
print(f"  {PASS if ok else FAIL} Skeleton → {skel_out.shape}  (expect (3,128,17))")

# dtype 확인
for name, arr in [("IMU", imu_out), ("Pressure", pres_out), ("Skeleton", skel_out)]:
    ok = arr.dtype == np.float32
    results.append((ok, f"{name} dtype float32", str(arr.dtype)))
    print(f"  {PASS if ok else FAIL} {name} dtype → {arr.dtype}  (expect float32)")


# ─── 요약 ────────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for ok, _, _ in results if ok)
failed = total - passed

print(f"\n{'='*60}")
if failed == 0:
    print(f"  \033[92m전체 통과: {passed}/{total}  — 전처리 파이프라인 배포 준비 완료\033[0m")
else:
    print(f"  \033[91m실패: {failed}/{total}  통과: {passed}/{total}\033[0m")
    print("\n  실패 항목:")
    for ok, name, detail in results:
        if not ok:
            print(f"    ✗ {name}: {detail}")
print(f"{'='*60}\n")

sys.exit(0 if failed == 0 else 1)
