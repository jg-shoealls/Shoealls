"""Raw IMU · Pressure · Skeleton → 13 base gait features.

특성 순서는 src/analysis/disease_classifier.py FEATURE_NAMES와 동일:
  [0]  gait_speed              m/s       — IMU 기반 보행 속도
  [1]  cadence                 steps/min — IMU 스텝 감지
  [2]  stride_regularity       0–1       — 자기상관 피크 비율
  [3]  step_symmetry           0–1       — L/R 간격 비율
  [4]  cop_sway                0–1       — 압력 무게중심 편심
  [5]  ml_variability          0–1       — ML 열 압력 변동
  [6]  heel_pressure_ratio     0–1       — 후족부 압력 비율
  [7]  forefoot_pressure_ratio 0–1       — 전족부+발끝 압력 비율
  [8]  arch_index              0–1       — 중족부 압력 비율
  [9]  pressure_asymmetry      0–1       — 내측/외측 압력 비대칭
  [10] acceleration_rms        m/s²      — 총 가속도 RMS
  [11] acceleration_variability 0–1     — 스트라이드간 가속도 CV
  [12] trunk_sway              deg/s     — 몸통 흔들림 (스켈레톤)

입력 규격:
  imu      : np.ndarray [T, 6] — raw (not normalized) m/s², rad/s
  pressure : np.ndarray [16, 8] or [T, 16, 8] — 0–1 normalized
  skeleton : np.ndarray [T, 17, 3] or None — COCO 관절 (optional)

샘플링 레이트: 기본 128 Hz (config에서 override 가능)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

SAMPLE_RATE  = 128      # Hz
_FEATURE_NAMES = [
    "gait_speed", "cadence", "stride_regularity", "step_symmetry",
    "cop_sway", "ml_variability", "heel_pressure_ratio", "forefoot_pressure_ratio",
    "arch_index", "pressure_asymmetry", "acceleration_rms", "acceleration_variability",
    "trunk_sway",
]

# disease_classifier.py FEATURE_NAMES와 순서 일치 여부를 임포트 시 검증
def _validate_feature_order() -> None:
    try:
        from src.analysis.disease_classifier import FEATURE_NAMES as _DC_NAMES
        if _FEATURE_NAMES != list(_DC_NAMES):
            raise RuntimeError(
                f"Feature order mismatch between feature_extractor and disease_classifier!\n"
                f"  extractor : {_FEATURE_NAMES}\n"
                f"  classifier: {list(_DC_NAMES)}"
            )
    except ImportError:
        pass  # 분석 모듈 없이 센서 레이어만 사용하는 경우

_validate_feature_order()


@dataclass
class GaitFeatures:
    gait_speed:               float = 1.2
    cadence:                  float = 115.0
    stride_regularity:        float = 0.85
    step_symmetry:            float = 0.92
    cop_sway:                 float = 0.04
    ml_variability:           float = 0.06
    heel_pressure_ratio:      float = 0.33
    forefoot_pressure_ratio:  float = 0.45
    arch_index:               float = 0.25
    pressure_asymmetry:       float = 0.05
    acceleration_rms:         float = 1.5
    acceleration_variability: float = 0.15
    trunk_sway:               float = 2.0

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in _FEATURE_NAMES], dtype=np.float32)


class GaitFeatureExtractor:
    """Raw 센서 배열에서 13개 보행 지표를 추출한다."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.fs = sample_rate

    def extract(
        self,
        imu: np.ndarray,
        pressure: np.ndarray,
        skeleton: np.ndarray | None = None,
    ) -> GaitFeatures:
        """
        Args:
            imu      : [T, 6]  — (ax,ay,az,gx,gy,gz) raw 단위
            pressure : [16,8] or [T,16,8] — 0–1 정규화 값
            skeleton : [T,17,3] COCO 관절 (없으면 기본값 사용)
        """
        imu = np.asarray(imu, dtype=np.float64)
        pressure = np.asarray(pressure, dtype=np.float64)
        if pressure.ndim == 3:
            p2d = pressure.mean(axis=0)
        else:
            p2d = pressure

        imu_f   = self._from_imu(imu)
        pres_f  = self._from_pressure(p2d)
        trunk   = self._trunk_sway(skeleton) if skeleton is not None else 2.0

        return GaitFeatures(
            gait_speed               = imu_f["gait_speed"],
            cadence                  = imu_f["cadence"],
            stride_regularity        = imu_f["stride_regularity"],
            step_symmetry            = imu_f["step_symmetry"],
            cop_sway                 = pres_f["cop_sway"],
            ml_variability           = pres_f["ml_variability"],
            heel_pressure_ratio      = pres_f["heel_pressure_ratio"],
            forefoot_pressure_ratio  = pres_f["forefoot_pressure_ratio"],
            arch_index               = pres_f["arch_index"],
            pressure_asymmetry       = pres_f["pressure_asymmetry"],
            acceleration_rms         = imu_f["acceleration_rms"],
            acceleration_variability = imu_f["acceleration_variability"],
            trunk_sway               = trunk,
        )

    # ── IMU ───────────────────────────────────────────────────────────────────

    def _from_imu(self, imu: np.ndarray) -> dict[str, float]:
        accel = imu[:, :3]                         # [T, 3]
        vert  = accel[:, 1]                        # y = 수직 (sagittal up)
        mag   = np.linalg.norm(accel, axis=1)      # [T] total acceleration

        steps = _detect_steps(vert, self.fs)
        n_steps = len(steps)
        dur_s   = len(vert) / self.fs

        # cadence
        cadence = float(np.clip(n_steps / dur_s * 60.0, 50.0, 200.0)) if dur_s > 0 else 115.0

        # stride regularity: autocorrelation peak at stride period
        stride_regularity = _autocorr_peak(vert, self.fs)

        # step symmetry: L/R interval proxy
        step_symmetry = _step_symmetry(steps)

        # gait speed: cadence × stride_length (Hausdorff linear model)
        stride_freq   = cadence / 120.0            # strides/s
        stride_length = 0.28 + 0.34 * stride_freq  # m (empirical)
        gait_speed    = float(np.clip(stride_length * stride_freq, 0.3, 2.5))

        # acceleration RMS
        acc_rms = float(np.sqrt(np.mean(mag ** 2)))

        # acceleration variability: stride-to-stride RMS CV
        if n_steps >= 2:
            stride_rms = np.array([
                np.sqrt(np.mean(mag[steps[i]: steps[i + 1]] ** 2))
                for i in range(n_steps - 1)
            ])
            mu = stride_rms.mean()
            acc_var = float(np.std(stride_rms) / (mu + 1e-9))
        else:
            acc_var = 0.15

        return {
            "gait_speed":               gait_speed,
            "cadence":                  cadence,
            "stride_regularity":        stride_regularity,
            "step_symmetry":            step_symmetry,
            "acceleration_rms":         float(np.clip(acc_rms, 0.0, 10.0)),
            "acceleration_variability": float(np.clip(acc_var, 0.0, 1.0)),
        }

    # ── Pressure ──────────────────────────────────────────────────────────────

    def _from_pressure(self, grid: np.ndarray) -> dict[str, float]:
        """16 × 8 압력 그리드 → 6 지표."""
        H, W = grid.shape
        total = grid.sum() or 1e-9

        # foot_zones.py와 동일한 구역 분할
        heel    = grid[11:, :]      # rows 11–15
        midfoot = grid[7:11, :]     # rows 7–10
        front   = grid[:7, :]       # rows 0–6 (forefoot + toes)

        h_sum = heel.sum()
        m_sum = midfoot.sum()
        f_sum = front.sum()

        heel_ratio     = float(h_sum / total)
        forefoot_ratio = float(f_sum / total)
        arch_denom     = h_sum + f_sum
        arch_index     = float(m_sum / arch_denom) if arch_denom > 1e-9 else 0.25

        # mediolateral asymmetry (cols 0–3 = medial, 4–7 = lateral)
        med = grid[:, :4].sum()
        lat = grid[:, 4:].sum()
        asym = float(abs(med - lat) / total)

        # Center of Pressure (CoP)
        rows = np.arange(H).reshape(-1, 1)
        cols = np.arange(W).reshape(1, -1)
        cop_r = float((grid * rows).sum() / total) / H
        cop_c = float((grid * cols).sum() / total) / W
        cop_sway = float(np.sqrt((cop_r - 0.5) ** 2 + (cop_c - 0.5) ** 2))

        # ML variability: normalised std of column-summed pressure
        col_p = grid.sum(axis=0)
        ml_var = float(np.std(col_p) / (col_p.mean() + 1e-9))

        return {
            "heel_pressure_ratio":     float(np.clip(heel_ratio, 0.0, 1.0)),
            "forefoot_pressure_ratio": float(np.clip(forefoot_ratio, 0.0, 1.0)),
            "arch_index":              float(np.clip(arch_index, 0.0, 1.0)),
            "pressure_asymmetry":      float(np.clip(asym, 0.0, 1.0)),
            "cop_sway":                float(np.clip(cop_sway, 0.0, 1.0)),
            "ml_variability":          float(np.clip(ml_var, 0.0, 1.0)),
        }

    # ── Skeleton ──────────────────────────────────────────────────────────────

    def _trunk_sway(self, skeleton: np.ndarray) -> float:
        """COCO 17관절 [T,17,3] → 몸통 흔들림 (deg/s)."""
        skel = np.asarray(skeleton, dtype=np.float64)
        if skel.ndim != 3 or skel.shape[1] < 13:
            return 2.0
        # COCO: 5=LShoulder, 6=RShoulder, 11=LHip, 12=RHip
        sh_mid = (skel[:, 5, :] + skel[:, 6, :]) / 2.0
        hi_mid = (skel[:, 11, :] + skel[:, 12, :]) / 2.0
        trunk  = sh_mid - hi_mid              # [T, 3]
        # lateral tilt: atan2(x, y)
        tilt   = np.degrees(np.arctan2(trunk[:, 0], trunk[:, 1]))
        sway   = float(np.std(tilt) * (self.fs / len(skel)))
        return float(np.clip(sway, 0.0, 20.0))


# ── 신호 처리 헬퍼 ────────────────────────────────────────────────────────────

def _detect_steps(vert: np.ndarray, fs: int, min_gap_ms: int = 250) -> list[int]:
    """수직 가속도에서 스텝 충격 피크 인덱스를 찾는다."""
    from scipy.signal import find_peaks
    min_gap = max(1, int(fs * min_gap_ms / 1000))
    mu, sd  = vert.mean(), vert.std()
    thresh  = mu + 0.3 * sd
    peaks, _ = find_peaks(vert, height=thresh, distance=min_gap)
    return list(peaks)


def _autocorr_peak(sig: np.ndarray, fs: int) -> float:
    """자기상관 함수의 보행 주기 피크 비율 (0–1).

    보행 주기 탐색 범위: 0.3–1.5 s.
    """
    n = len(sig)
    s = sig - sig.mean()
    denom = float(np.dot(s, s)) + 1e-9
    acf = np.correlate(s, s, mode="full")[n - 1:] / denom

    lo = int(0.3 * fs)
    hi = min(int(1.5 * fs), len(acf) - 1)
    if lo >= hi:
        return 0.7
    return float(np.clip(acf[lo:hi].max(), 0.0, 1.0))


def _step_symmetry(steps: list[int]) -> float:
    """홀/짝 스텝 간격 비율로 L/R 대칭도를 추정한다 (0–1)."""
    if len(steps) < 4:
        return 0.90
    ivl  = np.diff(steps, dtype=float)
    even = ivl[0::2]
    odd  = ivl[1::2]
    if len(even) == 0 or len(odd) == 0:
        return 0.90
    ratio = float(
        min(even.mean(), odd.mean()) / (max(even.mean(), odd.mean()) + 1e-9)
    )
    return float(np.clip(ratio, 0.0, 1.0))
