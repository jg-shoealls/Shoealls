"""보행 패턴 지표 추출기.

신발 내장 센서 (IMU, 족저압, 지자기, 기압) 로부터
임상적으로 의미 있는 보행 지표를 계산한다.

사용 예:
    extractor = GaitMetricsExtractor(fs=30)
    metrics = extractor.extract(sample_dict)
    # → {cadence_hz, stride_regularity, tremor_power, fog_index, foot_clearance_mean, ...}
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── 결과 컨테이너 ─────────────────────────────────────────────────────────────

@dataclass
class GaitMetrics:
    """센서별 보행 지표 모음."""

    # ── IMU 지표 ─────────────────
    cadence_hz:          float = 0.0  # 보행 주파수 (걸음/초)
    stride_regularity:   float = 0.0  # 보폭 규칙성 (0~1, 높을수록 규칙적)
    tremor_power_db:     float = 0.0  # 안정시 떨림 파워 (4~6 Hz 대역, dB)
    jerk_mean:           float = 0.0  # 평균 저크 크기 (가속도 변화율 — 낮을수록 부드러움)
    rms_acceleration:    float = 0.0  # RMS 가속도 (운동 강도)
    ml_sway:             float = 0.0  # 좌우(ML) 가속도 표준편차 (균형 불안정성)
    gait_smoothness:     float = 0.0  # SPARC: 스펙트럼 호 길이 (-값, 0에 가까울수록 부드러움)
    asymmetry_index:     float = 0.0  # 좌우 보행 비대칭 지수 (0~1)

    # ── 족저압 지표 ──────────────
    cop_path_length:     float = 0.0  # COP 궤적 총 길이 (압력 중심 이동)
    pressure_symmetry:   float = 0.0  # 좌우 압력 대칭성 (1=완전 대칭, 0=완전 비대칭)
    heel_toe_ratio:      float = 0.0  # 발뒤꿈치 / 앞발 압력 비율 (평발화시 증가)
    contact_area_cv:     float = 0.0  # 접촉 면적 변동계수 (보폭 일관성)
    push_off_power:      float = 0.0  # 앞발 압력 피크 파워 (추진력)

    # ── 지자기 지표 ──────────────
    fog_index:           float = 0.0  # 보행 동결 지수 (3~8 Hz / 전체 파워 비율)
    heading_stability:   float = 0.0  # 방위각 안정성 (1=안정, 0=불안정)
    turning_arc:         float = 0.0  # 방향 전환 호 (rad — 클수록 자연스러운 회전)

    # ── 기압(발 지상고) 지표 ──────
    foot_clearance_mean: float = 0.0  # 평균 발 지상고 (정규화, 높을수록 좋음)
    foot_clearance_cv:   float = 0.0  # 발 지상고 변동계수 (낮을수록 일관적)
    clearance_asymmetry: float = 0.0  # 지상고 좌우 비대칭 (0=대칭)
    step_height_decline: float = 0.0  # 연속 보행 중 지상고 감소율 (피로도)

    def to_dict(self) -> dict:
        return asdict(self)

    def stage_risk_score(self) -> float:
        """0~1 범위 종합 위험 점수 (높을수록 임상 단계 가능성)."""
        score = 0.0
        # 각 지표를 임계값 기반으로 정규화
        score += min(max((1.8 - self.cadence_hz) / 0.8, 0), 1) * 0.15
        score += min(max((0.95 - self.stride_regularity) / 0.45, 0), 1) * 0.15
        score += min(max(self.tremor_power_db / 20.0, 0), 1) * 0.15
        score += min(max(self.fog_index / 0.4, 0), 1) * 0.20
        score += min(max((1.0 - self.foot_clearance_mean) / 0.65, 0), 1) * 0.15
        score += min(max(self.asymmetry_index / 0.35, 0), 1) * 0.10
        score += min(max((1.0 - self.heading_stability) / 0.8, 0), 1) * 0.10
        return float(np.clip(score, 0, 1))


# ── 개별 지표 계산 함수 ───────────────────────────────────────────────────────

def _psd(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """단일 채널 신호의 파워 스펙트럼 밀도 (Welch 방식)."""
    from scipy.signal import welch
    f, pxx = welch(signal, fs=fs, nperseg=min(len(signal), 64))
    return f, pxx


def compute_cadence(imu: np.ndarray, fs: float) -> float:
    """수직 가속도 (채널 1) FFT에서 주 보행 주파수를 추출한다."""
    vertical = imu[1] if imu.shape[0] > 1 else imu[0]
    f, pxx = _psd(vertical, fs)
    # 보행 주파수 탐색 범위: 0.5~3.5 Hz
    mask = (f >= 0.5) & (f <= 3.5)
    if not mask.any():
        return 0.0
    return float(f[mask][np.argmax(pxx[mask])])


def compute_stride_regularity(imu: np.ndarray, fs: float) -> float:
    """자기상관 피크로 보폭 규칙성을 계산한다.

    1에 가까울수록 일정한 보폭; 파킨슨 진행 시 0.52까지 감소.
    """
    vertical = imu[1] if imu.shape[0] > 1 else imu[0]
    n = len(vertical)
    autocorr = np.correlate(vertical - vertical.mean(), vertical - vertical.mean(), mode="full")
    autocorr = autocorr[n - 1:]
    autocorr /= (autocorr[0] + 1e-8)

    # 첫 번째 주요 피크 탐색 (보폭 주기 해당)
    cadence_hz = compute_cadence(imu, fs)
    if cadence_hz < 0.1:
        return 0.0
    stride_lag = int(fs / cadence_hz)
    search_lo  = max(1, stride_lag - stride_lag // 3)
    search_hi  = min(n - 1, stride_lag + stride_lag // 3)
    if search_lo >= search_hi:
        return 0.0
    peak = float(autocorr[search_lo:search_hi].max())
    return float(np.clip(peak, 0, 1))


def compute_tremor_power(imu: np.ndarray, fs: float) -> float:
    """안정시 떨림 파워 (4~6 Hz 대역) — dB 단위."""
    resting_band = (4.0, 6.0)
    total_power  = 0.0
    tremor_power = 0.0
    for ch in range(min(3, imu.shape[0])):
        f, pxx = _psd(imu[ch], fs)
        total_power  += pxx.sum()
        mask = (f >= resting_band[0]) & (f <= resting_band[1])
        tremor_power += pxx[mask].sum()
    ratio = tremor_power / (total_power + 1e-10)
    return float(10 * np.log10(ratio + 1e-10))


def compute_jerk(imu: np.ndarray, fs: float) -> float:
    """평균 저크 크기: 가속도 시간 미분의 RMS."""
    accel = imu[:3] if imu.shape[0] >= 3 else imu
    jerk  = np.diff(accel, axis=1) * fs
    return float(np.sqrt((jerk ** 2).sum(axis=0)).mean())


def compute_gait_smoothness(imu: np.ndarray, fs: float, fc: float = 10.0) -> float:
    """SPARC (Spectral Arc Length): 스펙트럼 호 길이로 보행 부드러움 측정.

    Returns negative value — 0에 가까울수록 부드러운 보행.
    """
    from scipy.signal import butter, filtfilt
    accel_mag = np.sqrt((imu[:3] ** 2).sum(axis=0)) if imu.shape[0] >= 3 else np.abs(imu[0])
    b, a = butter(4, fc / (fs / 2), btype="low")
    try:
        smoothed = filtfilt(b, a, accel_mag)
    except Exception:
        smoothed = accel_mag

    n = len(smoothed)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    mag  = np.abs(np.fft.rfft(smoothed)) / n
    mag  = mag / (mag.max() + 1e-8)

    mask = freq <= fc
    dm   = np.diff(mag[mask])
    df   = np.diff(freq[mask])
    arc  = -np.sqrt(1 + (dm / (df + 1e-8)) ** 2).sum() * (df.mean() / fc)
    return float(arc)


def compute_asymmetry_index(imu: np.ndarray) -> float:
    """좌우(ML) 채널 비대칭 지수 (0=완전 대칭, 1=최대 비대칭)."""
    if imu.shape[0] < 2:
        return 0.0
    left  = np.abs(imu[0])
    right = np.abs(imu[1] if imu.shape[0] > 1 else imu[0])
    ai = np.abs(left - right) / (left + right + 1e-8)
    return float(ai.mean())


# ── 족저압 지표 ───────────────────────────────────────────────────────────────

def compute_cop_path_length(pressure: np.ndarray) -> float:
    """COP(압력 중심) 궤적 총 이동 거리."""
    T = pressure.shape[0]
    pressure_2d = pressure[:, 0] if pressure.ndim == 4 else pressure
    h, w = pressure_2d.shape[1], pressure_2d.shape[2]
    gy, gx = np.mgrid[0:h, 0:w].astype(float)

    cop_x, cop_y = [], []
    for t in range(T):
        frame = pressure_2d[t].astype(float)
        total = frame.sum() + 1e-8
        cop_x.append((frame * gx).sum() / total)
        cop_y.append((frame * gy).sum() / total)

    dx = np.diff(cop_x)
    dy = np.diff(cop_y)
    return float(np.sqrt(dx ** 2 + dy ** 2).sum())


def compute_pressure_symmetry(pressure: np.ndarray) -> float:
    """좌우 압력 대칭성 (1=완전 대칭)."""
    pressure_2d = pressure[:, 0] if pressure.ndim == 4 else pressure
    w = pressure_2d.shape[2]
    left  = pressure_2d[:, :, :w // 2].sum(axis=(1, 2))
    right = pressure_2d[:, :, w // 2:].sum(axis=(1, 2))
    sym   = 1 - np.abs(left - right) / (left + right + 1e-8)
    return float(sym.mean())


def compute_heel_toe_ratio(pressure: np.ndarray) -> float:
    """발뒤꿈치 / 앞발 압력 비율."""
    pressure_2d = pressure[:, 0] if pressure.ndim == 4 else pressure
    h = pressure_2d.shape[1]
    heel = pressure_2d[:, h // 2:, :].sum(axis=(1, 2))
    toe  = pressure_2d[:, :h // 3, :].sum(axis=(1, 2))
    return float((heel / (toe + 1e-8)).mean())


def compute_contact_area_cv(pressure: np.ndarray, threshold: float = 0.1) -> float:
    """접촉 면적 변동계수 (낮을수록 보폭 일관성 높음)."""
    pressure_2d = pressure[:, 0] if pressure.ndim == 4 else pressure
    area = (pressure_2d > threshold).sum(axis=(1, 2)).astype(float)
    cv = area.std() / (area.mean() + 1e-8)
    return float(cv)


def compute_push_off_power(pressure: np.ndarray) -> float:
    """앞발 압력 피크 파워 (추진력 지표)."""
    pressure_2d = pressure[:, 0] if pressure.ndim == 4 else pressure
    h = pressure_2d.shape[1]
    toe_pressure = pressure_2d[:, :h // 3, :].sum(axis=(1, 2))
    return float(toe_pressure.max())


# ── 지자기 지표 ───────────────────────────────────────────────────────────────

def compute_fog_index(mag_baro: np.ndarray, fs: float) -> float:
    """보행 동결(FOG) 지수: 3~8 Hz 파워 / 전체 파워 비율.

    mag_baro: (5, T) — 채널 0,1,2가 mx,my,mz
    0.15 이상이면 FOG 위험 구간.
    """
    fog_band   = (3.0, 8.0)
    total_power = 0.0
    fog_power   = 0.0
    for ch in range(3):
        f, pxx = _psd(mag_baro[ch], fs)
        total_power += pxx.sum()
        mask = (f >= fog_band[0]) & (f <= fog_band[1])
        fog_power += pxx[mask].sum()
    return float(fog_power / (total_power + 1e-10))


def compute_heading_stability(mag_baro: np.ndarray) -> float:
    """방위각 안정성 (1=안정, 0=불안정).

    mag_baro 채널 3은 heading (arctan2(my, mx)).
    """
    heading_ch = mag_baro[3] if mag_baro.shape[0] > 3 else mag_baro[0]
    # 헤딩 변화율의 표준편차로 안정성 측정
    d_heading = np.abs(np.diff(heading_ch))
    instability = d_heading.std() / (d_heading.mean() + 1e-8)
    return float(np.exp(-instability))


def compute_turning_arc(mag_baro: np.ndarray) -> float:
    """방향 전환 호 (rad) — 전체 방향 전환의 평균 크기."""
    mx = mag_baro[0]
    my = mag_baro[1]
    heading = np.arctan2(my, mx)
    d_heading = np.diff(np.unwrap(heading))
    return float(np.abs(d_heading).mean())


# ── 기압(발 지상고) 지표 ─────────────────────────────────────────────────────

def compute_foot_clearance(mag_baro: np.ndarray, fs: float) -> tuple[float, float]:
    """평균 발 지상고 및 변동계수.

    mag_baro 채널 4가 barometer (고도).
    Returns: (mean_clearance, cv)
    """
    baro = mag_baro[4] if mag_baro.shape[0] > 4 else mag_baro[-1]
    # 지상고 피크 추출 (유각기 최대값)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(baro, height=baro.max() * 0.3, distance=int(fs * 0.3))
    if len(peaks) < 2:
        return float(baro.max()), 1.0
    peak_heights = baro[peaks]
    mean_h = float(peak_heights.mean())
    cv     = float(peak_heights.std() / (mean_h + 1e-8))
    return mean_h, cv


def compute_step_height_decline(mag_baro: np.ndarray, fs: float) -> float:
    """연속 보행 중 지상고 감소율 (피로도 지표).

    전반부 vs 후반부 평균 발 지상고 차이.
    """
    baro = mag_baro[4] if mag_baro.shape[0] > 4 else mag_baro[-1]
    mid  = len(baro) // 2
    first_half  = baro[:mid].mean()
    second_half = baro[mid:].mean()
    decline = (first_half - second_half) / (first_half + 1e-8)
    return float(np.clip(decline, 0, 1))


# ── 통합 추출기 ───────────────────────────────────────────────────────────────

class GaitMetricsExtractor:
    """멀티모달 보행 데이터에서 전 지표를 한 번에 추출한다.

    Args:
        fs: 샘플링 주파수 (default: 30 Hz)
    """

    def __init__(self, fs: float = 30.0):
        self.fs = fs

    def extract(self, sample: dict) -> GaitMetrics:
        """
        Args:
            sample: {
                'imu':      numpy (6, T)  or torch tensor,
                'pressure': numpy (T, 1, H, W),
                'mag_baro': numpy (5, T)  — mag(4) + baro(1),
            }

        Returns:
            GaitMetrics dataclass
        """
        m = GaitMetrics()
        fs = self.fs

        def _np(x):
            return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

        # ── IMU 지표 ────────────────────────────────────────────────────────
        if "imu" in sample:
            imu = _np(sample["imu"])  # (6, T) or (C, T)

            m.cadence_hz        = compute_cadence(imu, fs)
            m.stride_regularity = compute_stride_regularity(imu, fs)
            m.tremor_power_db   = compute_tremor_power(imu, fs)
            m.jerk_mean         = compute_jerk(imu, fs)
            m.rms_acceleration  = float(np.sqrt((imu[:3] ** 2).sum(axis=0)).mean()) if imu.shape[0] >= 3 else 0.0
            m.ml_sway           = float(imu[0].std()) if imu.shape[0] >= 1 else 0.0
            m.gait_smoothness   = compute_gait_smoothness(imu, fs)
            m.asymmetry_index   = compute_asymmetry_index(imu)

        # ── 족저압 지표 ──────────────────────────────────────────────────────
        if "pressure" in sample:
            pressure = _np(sample["pressure"])  # (T, 1, H, W)

            m.cop_path_length  = compute_cop_path_length(pressure)
            m.pressure_symmetry = compute_pressure_symmetry(pressure)
            m.heel_toe_ratio   = compute_heel_toe_ratio(pressure)
            m.contact_area_cv  = compute_contact_area_cv(pressure)
            m.push_off_power   = compute_push_off_power(pressure)

        # ── 지자기 + 기압 지표 ────────────────────────────────────────────────
        if "mag_baro" in sample:
            mag_baro = _np(sample["mag_baro"])  # (5, T)

            m.fog_index        = compute_fog_index(mag_baro, fs)
            m.heading_stability = compute_heading_stability(mag_baro)
            m.turning_arc      = compute_turning_arc(mag_baro)

            mean_c, cv_c       = compute_foot_clearance(mag_baro, fs)
            m.foot_clearance_mean = mean_c
            m.foot_clearance_cv   = cv_c
            m.step_height_decline = compute_step_height_decline(mag_baro, fs)

        return m

    def extract_batch(self, dataset, indices: Optional[list] = None) -> list[GaitMetrics]:
        """데이터셋의 여러 샘플을 일괄 처리한다."""
        if indices is None:
            indices = range(len(dataset))
        return [self.extract(dataset[i]) for i in indices]

    def stage_summary(self, dataset, labels: np.ndarray) -> dict:
        """단계별 지표 평균값 요약표를 반환한다."""
        from collections import defaultdict
        stage_metrics: dict = defaultdict(list)

        for i in range(len(dataset)):
            m = self.extract(dataset[i])
            stage = int(labels[i])
            stage_metrics[stage].append(m.to_dict())

        summary = {}
        for stage, metric_list in stage_metrics.items():
            keys = metric_list[0].keys()
            summary[stage] = {
                k: float(np.mean([m[k] for m in metric_list]))
                for k in keys
            }
        return summary
