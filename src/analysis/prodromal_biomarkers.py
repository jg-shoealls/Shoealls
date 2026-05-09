"""전임상 파킨슨 바이오마커 패널.

발병 5~10년 전 나타나는 비운동 전임상 신호 3종을 계산한다:
  1. 보행 변동성 (stride-to-stride CV, step width CV 등)
  2. REM 수면 장애 대리 지표 (정적 구간 4-6 Hz 트레모 + 움직임 자기상관)
  3. 후각 선별 점수 (외부 검사 입력)

MDS 전임상 기준 (Berg et al. 2015) 기반 복합 점수 산출.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────

@dataclass
class ProdromalBiomarker:
    name: str
    korean_name: str
    value: float
    normal_range: tuple[float, float]   # (low, high) 정상 범위
    unit: str
    sensor: str                          # "IMU" | "PRESSURE" | "EXTERNAL" | "FUSION"
    clinical_meaning: str
    is_abnormal: bool = field(init=False)

    def __post_init__(self):
        lo, hi = self.normal_range
        self.is_abnormal = not (lo <= self.value <= hi)


@dataclass
class ProdromalPanel:
    biomarkers: list[ProdromalBiomarker]
    composite_score: float       # 0–1, 높을수록 전임상 위험
    prodrome_stage: str          # "정상" | "전임상" | "프로드로말" | "발현"
    stage_confidence: float      # 0–1
    abnormal_count: int = field(init=False)

    def __post_init__(self):
        self.abnormal_count = sum(b.is_abnormal for b in self.biomarkers)

    def summary(self) -> str:
        lines = [
            f"전임상 단계: {self.prodrome_stage}  (복합점수: {self.composite_score:.3f}, 신뢰도: {self.stage_confidence:.2f})",
            f"이상 바이오마커: {self.abnormal_count}/{len(self.biomarkers)}개",
        ]
        for b in self.biomarkers:
            flag = "⚠" if b.is_abnormal else "✓"
            lines.append(f"  {flag} {b.korean_name}: {b.value:.4f} {b.unit}  (정상 {b.normal_range[0]}–{b.normal_range[1]})")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 병기 분류 임계값 (MDS prodromal criteria)
# ─────────────────────────────────────────────
_STAGE_THRESHOLDS = (0.15, 0.30, 0.55)
_STAGE_LABELS = ("정상", "전임상", "프로드로말", "발현")

# 복합 점수 가중치 (MDS 특이도 기준)
_W_OLFACTORY  = 0.40
_W_REM_TREMOR = 0.30
_W_VARIABILITY = 0.30


# ─────────────────────────────────────────────
# 내부 신호 처리 유틸
# ─────────────────────────────────────────────

def _coefficient_of_variation(arr: np.ndarray) -> float:
    """CV = std/mean. 빈 배열이나 평균=0이면 0 반환."""
    if arr.size < 2:
        return 0.0
    m = float(np.mean(arr))
    return float(np.std(arr) / m) if abs(m) > 1e-9 else 0.0


def _find_peaks_simple(signal: np.ndarray, min_distance: int = 10) -> np.ndarray:
    """최소 간격 조건을 만족하는 극대값 인덱스 반환."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def _bandpower(signal: np.ndarray, fs: float, f_low: float, f_high: float) -> float:
    """신호에서 [f_low, f_high] Hz 대역 파워 비율을 반환한다."""
    n = len(signal)
    if n < 8:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(signal)) ** 2
    total = float(psd.sum())
    if total < 1e-12:
        return 0.0
    band_mask = (freqs >= f_low) & (freqs <= f_high)
    return float(psd[band_mask].sum() / total)


# ─────────────────────────────────────────────
# 바이오마커 계산 함수
# ─────────────────────────────────────────────

def _stride_time_cv(imu: np.ndarray, fs: float = 100.0) -> float:
    """보폭 시간 변동계수 (CV).

    수직 가속도(Z, index 2) 피크 간격의 CV.
    정상: <0.03, 전임상 PD: 0.03–0.08, 파킨슨: >0.08.
    """
    accel_z = imu[2] if imu.shape[0] > 2 else imu[0]
    accel_z = accel_z - float(np.mean(accel_z))
    peaks = _find_peaks_simple(accel_z, min_distance=int(fs * 0.3))
    if len(peaks) < 3:
        return 0.0
    intervals = np.diff(peaks) / fs   # 초 단위
    return _coefficient_of_variation(intervals)


def _step_width_cv(imu: np.ndarray, fs: float = 100.0) -> float:
    """보폭 너비 변동계수 (CV).

    내외측 가속도(X, index 0) 양의 피크값의 CV.
    정상: <0.15.
    """
    accel_x = imu[0] if imu.shape[0] > 0 else imu[0]
    accel_x = accel_x - float(np.mean(accel_x))
    peaks = _find_peaks_simple(accel_x, min_distance=int(fs * 0.25))
    if len(peaks) < 3:
        return 0.0
    return _coefficient_of_variation(np.abs(accel_x[peaks]))


def _cadence_variability(imu: np.ndarray, fs: float = 100.0) -> float:
    """보행 리듬 변동성: 순간 보행률(steps/s)의 SD.

    정상: <0.05 steps/s.
    """
    accel_z = imu[2] if imu.shape[0] > 2 else imu[0]
    accel_z = accel_z - float(np.mean(accel_z))
    peaks = _find_peaks_simple(accel_z, min_distance=int(fs * 0.3))
    if len(peaks) < 3:
        return 0.0
    intervals = np.diff(peaks) / fs
    instantaneous_cadence = 1.0 / (intervals + 1e-9)
    return float(np.std(instantaneous_cadence))


def _double_support_ratio(pressure: np.ndarray) -> float:
    """양발 지지 시간 비율.

    pressure: (T, H, W) 또는 (T, H*W).
    좌우를 상하로 분리해 동시 지지 프레임 비율을 구한다.
    정상 보행: 0.18–0.26, PD에서 증가.
    """
    if pressure.ndim == 3:
        p = pressure
    elif pressure.ndim == 4:
        p = pressure[:, 0]           # (T, H, W)
    else:
        return 0.22                  # 분석 불가 → 정상값 반환

    T, H, W = p.shape
    mid = H // 2
    left_active  = p[:, :mid, :].sum(axis=(1, 2)) > 0.01
    right_active = p[:, mid:, :].sum(axis=(1, 2)) > 0.01
    both = (left_active & right_active).sum()
    return float(both) / max(T, 1)


def _rest_tremor_index(imu: np.ndarray, fs: float = 100.0) -> float:
    """안정 시 4–6 Hz 트레모 지수 (REM 수면 장애 대리 지표).

    정적 구간(RMS 하위 25%)에서 각 가속도 채널의 4–6 Hz 대역 파워비 평균.
    sqrt(x²+y²+z²)는 항상 양수이므로 FFT 주파수가 2배로 올라가는 문제를 피하기 위해
    개별 채널에서 직접 밴드파워를 계산한다.

    정상: <0.10, 전임상 PD: 0.10–0.25, 파킨슨: >0.25.
    """
    n_ch = min(3, imu.shape[0])
    accel = imu[:n_ch]                          # (n_ch, T)
    accel_mag = np.sqrt((accel ** 2).sum(axis=0))  # 정적 구간 탐색용

    window = int(fs * 1.0)          # 1초 윈도우
    if accel_mag.shape[0] < window * 2:
        # 신호가 짧으면 전체 채널에 직접 적용
        ratios = [_bandpower(accel[ch], fs, 4.0, 6.0) for ch in range(n_ch)]
        return float(np.mean(ratios))

    step = window // 2
    rms_per_window = [
        float(np.sqrt(np.mean(accel_mag[i: i + window] ** 2)))
        for i in range(0, len(accel_mag) - window, step)
    ]
    threshold = float(np.percentile(rms_per_window, 25))

    quiet_per_ch: list[list[np.ndarray]] = [[] for _ in range(n_ch)]
    for idx, rms in enumerate(rms_per_window):
        if rms <= threshold:
            start = idx * step
            for ch in range(n_ch):
                quiet_per_ch[ch].append(accel[ch][start: start + window])

    if not quiet_per_ch[0]:
        return 0.0
    ratios = [
        _bandpower(np.concatenate(quiet_per_ch[ch]), fs, 4.0, 6.0)
        for ch in range(n_ch)
    ]
    return float(np.mean(ratios))


def _nocturnal_movement_regularity(imu: np.ndarray, fs: float = 100.0) -> float:
    """움직임 규칙성 (자기상관 기반 REM 대리 지표).

    야간 IMU가 없을 때 세션 IMU 자기상관으로 대체.
    높은 규칙성(→1.0)이 정상; 파킨슨 전임상에서 불규칙성 증가.
    정상: 0.60–1.00.
    """
    accel_mag = np.sqrt((imu[:3] ** 2).sum(axis=0))
    sig = accel_mag - float(np.mean(accel_mag))
    n = len(sig)
    if n < 20:
        return 0.8
    lag = min(int(fs * 1.0), n // 4)   # 1초 lag
    denom = float(np.dot(sig, sig))
    if denom < 1e-12:
        return 1.0
    corr = float(np.dot(sig[lag:], sig[: n - lag])) / denom
    return float(np.clip(corr, 0.0, 1.0))


# ─────────────────────────────────────────────
# 메인 추출기
# ─────────────────────────────────────────────

class ProdrOmalBiomarkerExtractor:
    """IMU + 족압 시퀀스에서 전임상 파킨슨 바이오마커 패널을 계산한다.

    Args:
        fs: IMU 샘플링 레이트 (Hz). 기본 100 Hz.
    """

    def __init__(self, fs: float = 100.0):
        self.fs = fs

    def extract(
        self,
        imu_seq: np.ndarray,          # (C, T), C≥3 (최소 3축 가속도)
        pressure_seq: np.ndarray,     # (T, H, W) | (T, 1, H, W) | (T, H*W)
        external: dict | None = None, # {"olfactory_screen_score": 0.85}
    ) -> ProdromalPanel:
        """바이오마커 패널을 계산하고 전임상 단계를 분류한다."""

        ext = external or {}
        fs = self.fs

        # ── 1. 보행 변동성 군 ────────────────────────────────────
        stcv  = _stride_time_cv(imu_seq, fs)
        swcv  = _step_width_cv(imu_seq, fs)
        cadv  = _cadence_variability(imu_seq, fs)

        # pressure_seq 형태 정규화
        if pressure_seq.ndim == 4:
            pres = pressure_seq[:, 0]    # (T, H, W)
        elif pressure_seq.ndim == 2:
            T = pressure_seq.shape[0]
            side = int(np.sqrt(pressure_seq.shape[1]))
            pres = pressure_seq.reshape(T, side, side) if side * side == pressure_seq.shape[1] else pressure_seq[:, :, None]
        else:
            pres = pressure_seq

        dsr = _double_support_ratio(pres)

        # ── 2. REM 수면 장애 대리 지표 군 ────────────────────────
        rti  = _rest_tremor_index(imu_seq, fs)
        nmr  = _nocturnal_movement_regularity(imu_seq, fs)

        # ── 3. 후각 선별 점수 ─────────────────────────────────────
        olfactory = float(ext.get("olfactory_screen_score", 1.0))
        olfactory = float(np.clip(olfactory, 0.0, 1.0))

        # ── 복합 점수 계산 ─────────────────────────────────────────
        # 각 지표를 [0,1] 위험 스케일로 정규화 후 가중 결합
        risk_variability = float(np.clip(stcv / 0.08, 0.0, 1.0))   # 0.08이 위험 상한
        risk_rem         = float(np.clip(rti  / 0.30, 0.0, 1.0))   # 0.30이 위험 상한
        risk_olfactory   = float(np.clip(1.0 - olfactory, 0.0, 1.0))  # 낮을수록 위험

        composite = (
            _W_VARIABILITY * risk_variability
            + _W_REM_TREMOR  * risk_rem
            + _W_OLFACTORY   * risk_olfactory
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        # ── 병기 분류 ──────────────────────────────────────────────
        t1, t2, t3 = _STAGE_THRESHOLDS
        if composite < t1:
            stage, conf = _STAGE_LABELS[0], 1.0 - composite / t1
        elif composite < t2:
            stage, conf = _STAGE_LABELS[1], (composite - t1) / (t2 - t1)
        elif composite < t3:
            stage, conf = _STAGE_LABELS[2], (composite - t2) / (t3 - t2)
        else:
            stage, conf = _STAGE_LABELS[3], min(1.0, (composite - t3) / (1.0 - t3 + 1e-9))
        conf = float(np.clip(conf, 0.0, 1.0))

        # ── 바이오마커 목록 조립 ──────────────────────────────────
        biomarkers = [
            ProdromalBiomarker(
                name="stride_time_cv",
                korean_name="보폭 시간 변동계수",
                value=round(stcv, 5),
                normal_range=(0.00, 0.03),
                unit="CV",
                sensor="IMU",
                clinical_meaning="보폭 시간 규칙성; 증가 시 소뇌·기저핵 미세 제어 저하 시사",
            ),
            ProdromalBiomarker(
                name="step_width_cv",
                korean_name="보폭 너비 변동계수",
                value=round(swcv, 5),
                normal_range=(0.00, 0.15),
                unit="CV",
                sensor="IMU",
                clinical_meaning="내외측 안정성; 증가 시 전정-소뇌 경로 이상",
            ),
            ProdromalBiomarker(
                name="cadence_variability",
                korean_name="보행 리듬 변동성",
                value=round(cadv, 5),
                normal_range=(0.00, 0.05),
                unit="steps/s SD",
                sensor="IMU",
                clinical_meaning="보행 리듬 일관성; 전임상 파킨슨에서 경미하게 증가",
            ),
            ProdromalBiomarker(
                name="double_support_ratio",
                korean_name="양발 지지 시간 비율",
                value=round(dsr, 5),
                normal_range=(0.18, 0.26),
                unit="ratio",
                sensor="PRESSURE",
                clinical_meaning="양발 지지 증가 시 보행 불안정·파킨슨 초기 징후",
            ),
            ProdromalBiomarker(
                name="rest_tremor_index",
                korean_name="안정 시 트레모 지수",
                value=round(rti, 5),
                normal_range=(0.00, 0.10),
                unit="power ratio",
                sensor="IMU",
                clinical_meaning="정적 구간 4-6 Hz 진동; REM 수면 장애 대리 지표, 파킨슨 전임상 마커",
            ),
            ProdromalBiomarker(
                name="nocturnal_movement_regularity",
                korean_name="움직임 규칙성 지수",
                value=round(nmr, 5),
                normal_range=(0.60, 1.00),
                unit="autocorr",
                clinical_meaning="움직임 패턴 규칙성; 감소 시 RBD(REM 수면 행동 장애) 시사",
                sensor="IMU",
            ),
            ProdromalBiomarker(
                name="olfactory_screen_score",
                korean_name="후각 선별 점수",
                value=round(olfactory, 5),
                normal_range=(0.70, 1.00),
                unit="score",
                sensor="EXTERNAL",
                clinical_meaning="후각 저하는 파킨슨 발병 4-6년 전 가장 이른 전임상 마커",
            ),
            ProdromalBiomarker(
                name="prodromal_composite_score",
                korean_name="전임상 복합 점수",
                value=round(composite, 5),
                normal_range=(0.00, 0.15),
                unit="score",
                sensor="FUSION",
                clinical_meaning="MDS 기준 가중 복합 점수 (후각 0.4 / REM 대리 0.3 / 변동성 0.3)",
            ),
        ]

        return ProdromalPanel(
            biomarkers=biomarkers,
            composite_score=composite,
            prodrome_stage=stage,
            stage_confidence=conf,
        )
