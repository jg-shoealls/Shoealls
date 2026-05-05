"""보행 이벤트 자동 감지 및 구간 분할 모듈.

IMU 가속도 피크와 족저압 접지 패턴을 활용하여
힐 스트라이크(Heel Strike), 토 오프(Toe Off) 이벤트를 감지하고
보행 주기를 자동으로 분할합니다.

Gait event detection and segmentation module.
Detects heel strike and toe off events from IMU acceleration peaks
and pressure ground contact patterns, then segments gait cycles.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal


@dataclass
class GaitEvent:
    """단일 보행 이벤트."""

    event_type: str  # "heel_strike" or "toe_off"
    timestamp_idx: int  # 샘플 인덱스
    confidence: float  # 0~1
    side: str = "unknown"  # "left", "right", "unknown"


@dataclass
class GaitCycle:
    """단일 보행 주기 (heel strike ~ next heel strike)."""

    start_idx: int
    end_idx: int
    heel_strike_idx: int
    toe_off_idx: int
    stance_duration: int  # 프레임 수
    swing_duration: int
    side: str = "unknown"


@dataclass
class GaitParameters:
    """보행 파라미터 요약."""

    num_cycles: int = 0
    mean_stride_time: float = 0.0
    std_stride_time: float = 0.0
    mean_cadence: float = 0.0  # steps/min
    mean_stance_ratio: float = 0.0
    mean_swing_ratio: float = 0.0
    stride_variability: float = 0.0  # CV of stride time
    symmetry_index: float = 0.0  # left-right symmetry
    events: list = field(default_factory=list)
    cycles: list = field(default_factory=list)


class IMUEventDetector:
    """IMU 가속도 기반 보행 이벤트 감지.

    수직 가속도(vertical acceleration)의 피크를 분석하여
    힐 스트라이크와 토 오프를 감지합니다.
    """

    def __init__(
        self,
        sampling_rate: float = 100.0,
        min_peak_distance_ms: float = 300.0,
        heel_strike_prominence: float = 0.5,
        toe_off_prominence: float = 0.3,
        lowpass_cutoff: float = 20.0,
    ):
        self.sampling_rate = sampling_rate
        self.min_peak_distance = int(min_peak_distance_ms / 1000.0 * sampling_rate)
        self.heel_strike_prominence = heel_strike_prominence
        self.toe_off_prominence = toe_off_prominence
        self.lowpass_cutoff = lowpass_cutoff

    def _lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """버터워스 저역 통과 필터."""
        nyquist = self.sampling_rate / 2.0
        if self.lowpass_cutoff >= nyquist:
            return data
        b, a = signal.butter(4, self.lowpass_cutoff / nyquist, btype="low")
        return signal.filtfilt(b, a, data, axis=0)

    def detect(
        self,
        imu_data: np.ndarray,
        vertical_axis: int = 2,
    ) -> list[GaitEvent]:
        """IMU 데이터에서 보행 이벤트 감지.

        Args:
            imu_data: (T, C) IMU 가속도 데이터. C >= 3 (ax, ay, az).
            vertical_axis: 수직 축 인덱스 (기본 z=2).

        Returns:
            감지된 GaitEvent 리스트 (시간순 정렬).
        """
        if imu_data.ndim != 2 or imu_data.shape[1] < 3:
            raise ValueError(f"Expected (T, C>=3) array, got {imu_data.shape}")

        acc_vertical = imu_data[:, vertical_axis].copy()
        acc_filtered = self._lowpass_filter(acc_vertical)

        events = []

        # Heel strike: 수직 가속도의 양의 피크 (발이 지면에 닿는 충격)
        hs_peaks, hs_props = signal.find_peaks(
            acc_filtered,
            distance=self.min_peak_distance,
            prominence=self.heel_strike_prominence,
        )
        for idx, peak_idx in enumerate(hs_peaks):
            prominence = hs_props["prominences"][idx]
            max_prom = max(hs_props["prominences"]) if len(hs_props["prominences"]) > 0 else 1.0
            confidence = min(prominence / (max_prom + 1e-8), 1.0)
            events.append(GaitEvent(
                event_type="heel_strike",
                timestamp_idx=int(peak_idx),
                confidence=float(confidence),
            ))

        # Toe off: 수직 가속도의 음의 피크 (발이 지면을 떠나는 순간)
        to_peaks, to_props = signal.find_peaks(
            -acc_filtered,
            distance=self.min_peak_distance,
            prominence=self.toe_off_prominence,
        )
        for idx, peak_idx in enumerate(to_peaks):
            prominence = to_props["prominences"][idx]
            max_prom = max(to_props["prominences"]) if len(to_props["prominences"]) > 0 else 1.0
            confidence = min(prominence / (max_prom + 1e-8), 1.0)
            events.append(GaitEvent(
                event_type="toe_off",
                timestamp_idx=int(peak_idx),
                confidence=float(confidence),
            ))

        events.sort(key=lambda e: e.timestamp_idx)
        return events


class PressureEventDetector:
    """족저압 기반 보행 이벤트 감지.

    압력 센서 총합의 접지/비접지 전환점에서 이벤트를 감지합니다.
    """

    def __init__(
        self,
        contact_threshold: float = 0.1,
        min_contact_frames: int = 10,
    ):
        self.contact_threshold = contact_threshold
        self.min_contact_frames = min_contact_frames

    def detect(self, pressure_data: np.ndarray) -> list[GaitEvent]:
        """족저압 데이터에서 보행 이벤트 감지.

        Args:
            pressure_data: (T, H, W) 또는 (T, N) 압력 데이터.

        Returns:
            감지된 GaitEvent 리스트.
        """
        if pressure_data.ndim == 3:
            # (T, H, W) -> 각 프레임의 총 압력
            total_pressure = pressure_data.reshape(pressure_data.shape[0], -1).sum(axis=1)
        elif pressure_data.ndim == 2:
            total_pressure = pressure_data.sum(axis=1)
        else:
            raise ValueError(f"Expected (T, H, W) or (T, N) array, got {pressure_data.shape}")

        # 접지 여부 이진 신호
        is_contact = total_pressure > self.contact_threshold
        events = []

        # 상태 전환 감지
        for i in range(1, len(is_contact)):
            if is_contact[i] and not is_contact[i - 1]:
                # 비접지 → 접지 = heel strike
                # 최소 접지 프레임 확인
                end = min(i + self.min_contact_frames, len(is_contact))
                if np.all(is_contact[i:end]):
                    confidence = min(total_pressure[i] / (total_pressure.max() + 1e-8), 1.0)
                    events.append(GaitEvent(
                        event_type="heel_strike",
                        timestamp_idx=i,
                        confidence=float(confidence),
                    ))

            elif not is_contact[i] and is_contact[i - 1]:
                # 접지 → 비접지 = toe off
                start = max(i - self.min_contact_frames, 0)
                if np.all(is_contact[start:i]):
                    confidence = min(total_pressure[i - 1] / (total_pressure.max() + 1e-8), 1.0)
                    events.append(GaitEvent(
                        event_type="toe_off",
                        timestamp_idx=i,
                        confidence=float(confidence),
                    ))

        return events


class GaitSegmenter:
    """보행 주기 분할 및 파라미터 계산.

    감지된 이벤트를 기반으로 보행 주기를 분할하고
    보행 파라미터를 계산합니다.
    """

    def __init__(self, sampling_rate: float = 100.0):
        self.sampling_rate = sampling_rate

    def segment_cycles(self, events: list[GaitEvent]) -> list[GaitCycle]:
        """이벤트 목록에서 보행 주기 추출.

        힐 스트라이크 ~ 다음 힐 스트라이크를 한 주기로 정의.

        Args:
            events: 시간순 정렬된 GaitEvent 리스트.

        Returns:
            GaitCycle 리스트.
        """
        heel_strikes = [e for e in events if e.event_type == "heel_strike"]
        toe_offs = [e for e in events if e.event_type == "toe_off"]

        if len(heel_strikes) < 2:
            return []

        cycles = []
        for i in range(len(heel_strikes) - 1):
            hs_current = heel_strikes[i]
            hs_next = heel_strikes[i + 1]

            # 이 주기 내의 toe off 찾기
            cycle_toe_offs = [
                to for to in toe_offs
                if hs_current.timestamp_idx < to.timestamp_idx < hs_next.timestamp_idx
            ]

            if not cycle_toe_offs:
                continue

            to_event = cycle_toe_offs[0]
            stance = to_event.timestamp_idx - hs_current.timestamp_idx
            swing = hs_next.timestamp_idx - to_event.timestamp_idx

            cycles.append(GaitCycle(
                start_idx=hs_current.timestamp_idx,
                end_idx=hs_next.timestamp_idx,
                heel_strike_idx=hs_current.timestamp_idx,
                toe_off_idx=to_event.timestamp_idx,
                stance_duration=stance,
                swing_duration=swing,
                side=hs_current.side,
            ))

        return cycles

    def compute_parameters(self, cycles: list[GaitCycle]) -> GaitParameters:
        """보행 주기에서 파라미터 계산.

        Args:
            cycles: GaitCycle 리스트.

        Returns:
            GaitParameters 요약.
        """
        if not cycles:
            return GaitParameters()

        stride_times = np.array([
            (c.end_idx - c.start_idx) / self.sampling_rate for c in cycles
        ])
        stance_ratios = np.array([
            c.stance_duration / (c.stance_duration + c.swing_duration)
            for c in cycles
        ])
        swing_ratios = 1.0 - stance_ratios

        mean_stride = float(np.mean(stride_times))
        cadence = 60.0 / mean_stride if mean_stride > 0 else 0.0

        # 좌우 대칭성 (좌/우 정보가 있는 경우)
        left_strides = [s for c, s in zip(cycles, stride_times) if c.side == "left"]
        right_strides = [s for c, s in zip(cycles, stride_times) if c.side == "right"]
        if left_strides and right_strides:
            mean_left = np.mean(left_strides)
            mean_right = np.mean(right_strides)
            symmetry = 1.0 - abs(mean_left - mean_right) / (0.5 * (mean_left + mean_right) + 1e-8)
        else:
            symmetry = 1.0  # 좌우 구분 없으면 완전 대칭으로 가정

        return GaitParameters(
            num_cycles=len(cycles),
            mean_stride_time=mean_stride,
            std_stride_time=float(np.std(stride_times)),
            mean_cadence=cadence,
            mean_stance_ratio=float(np.mean(stance_ratios)),
            mean_swing_ratio=float(np.mean(swing_ratios)),
            stride_variability=float(np.std(stride_times) / (mean_stride + 1e-8)),
            symmetry_index=float(symmetry),
            cycles=cycles,
        )


def detect_and_segment(
    imu_data: Optional[np.ndarray] = None,
    pressure_data: Optional[np.ndarray] = None,
    sampling_rate: float = 100.0,
    merge_window: int = 5,
) -> GaitParameters:
    """IMU와 족저압 데이터를 융합하여 보행 이벤트 감지 및 분할.

    두 센서의 이벤트를 융합하여 더 정확한 감지를 수행합니다.

    Args:
        imu_data: (T, C) IMU 가속도 데이터 (선택).
        pressure_data: (T, H, W) 또는 (T, N) 압력 데이터 (선택).
        sampling_rate: 샘플링 레이트 (Hz).
        merge_window: 이벤트 병합 윈도우 (프레임).

    Returns:
        GaitParameters 보행 파라미터.
    """
    if imu_data is None and pressure_data is None:
        raise ValueError("At least one of imu_data or pressure_data must be provided")

    all_events: list[GaitEvent] = []

    if imu_data is not None:
        imu_detector = IMUEventDetector(sampling_rate=sampling_rate)
        imu_events = imu_detector.detect(imu_data)
        all_events.extend(imu_events)

    if pressure_data is not None:
        pressure_detector = PressureEventDetector()
        pressure_events = pressure_detector.detect(pressure_data)
        all_events.extend(pressure_events)

    # 두 센서 이벤트 병합 (가까운 이벤트는 하나로)
    if imu_data is not None and pressure_data is not None:
        all_events = _merge_events(all_events, merge_window)

    segmenter = GaitSegmenter(sampling_rate=sampling_rate)
    cycles = segmenter.segment_cycles(all_events)
    params = segmenter.compute_parameters(cycles)
    params.events = all_events
    return params


def _merge_events(events: list[GaitEvent], window: int) -> list[GaitEvent]:
    """가까운 동일 유형 이벤트를 병합."""
    if not events:
        return events

    events.sort(key=lambda e: (e.event_type, e.timestamp_idx))
    merged = []

    for event_type in ("heel_strike", "toe_off"):
        typed = [e for e in events if e.event_type == event_type]
        if not typed:
            continue

        group = [typed[0]]
        for e in typed[1:]:
            if e.timestamp_idx - group[-1].timestamp_idx <= window:
                group.append(e)
            else:
                merged.append(_best_event(group))
                group = [e]
        merged.append(_best_event(group))

    merged.sort(key=lambda e: e.timestamp_idx)
    return merged


def _best_event(group: list[GaitEvent]) -> GaitEvent:
    """그룹 내 가장 신뢰도 높은 이벤트 반환."""
    return max(group, key=lambda e: e.confidence)


def plot_gait_events(
    params: GaitParameters,
    imu_data: Optional[np.ndarray] = None,
    pressure_data: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """보행 이벤트 및 주기 시각화.

    Args:
        params: detect_and_segment() 출력.
        imu_data: (T, C) IMU 데이터 (시각화용).
        pressure_data: (T, H, W) 또는 (T, N) 압력 데이터.
        save_path: 저장 경로.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    num_plots = sum([imu_data is not None, pressure_data is not None, len(params.cycles) > 0])
    if num_plots == 0:
        return

    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4 * num_plots), squeeze=False)
    axes = axes.flatten()
    plot_idx = 0

    # IMU 가속도 + 이벤트
    if imu_data is not None:
        ax = axes[plot_idx]
        T = imu_data.shape[0]
        t = np.arange(T)

        if imu_data.shape[1] >= 3:
            ax.plot(t, imu_data[:, 2], label="Vertical Acc", color="#2196F3", alpha=0.7)
        ax.set_ylabel("Acceleration")
        ax.set_title("IMU Vertical Acceleration + Gait Events")

        for event in params.events:
            color = "#F44336" if event.event_type == "heel_strike" else "#4CAF50"
            marker = "v" if event.event_type == "heel_strike" else "^"
            label = event.event_type.replace("_", " ").title()
            ax.axvline(event.timestamp_idx, color=color, alpha=0.3, linestyle="--")
            if imu_data.shape[1] >= 3:
                ax.plot(event.timestamp_idx, imu_data[event.timestamp_idx, 2],
                        marker=marker, color=color, markersize=8)

        # 범례 중복 제거
        from collections import OrderedDict
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(OrderedDict(zip(labels, handles)).values(),
                  OrderedDict(zip(labels, handles)).keys(), loc="upper right")
        ax.set_xlabel("Frame")
        plot_idx += 1

    # 족저압 총합 + 이벤트
    if pressure_data is not None:
        ax = axes[plot_idx]
        if pressure_data.ndim == 3:
            total = pressure_data.reshape(pressure_data.shape[0], -1).sum(axis=1)
        else:
            total = pressure_data.sum(axis=1)

        t = np.arange(len(total))
        ax.fill_between(t, total, alpha=0.3, color="#FF9800")
        ax.plot(t, total, color="#FF9800", label="Total Pressure")
        ax.set_ylabel("Total Pressure")
        ax.set_title("Foot Pressure + Gait Events")

        for event in params.events:
            color = "#F44336" if event.event_type == "heel_strike" else "#4CAF50"
            ax.axvline(event.timestamp_idx, color=color, alpha=0.3, linestyle="--")

        ax.legend(loc="upper right")
        ax.set_xlabel("Frame")
        plot_idx += 1

    # 보행 주기 파라미터 바 차트
    if params.cycles:
        ax = axes[plot_idx]
        cycle_indices = np.arange(len(params.cycles))
        stance_ratios = [c.stance_duration / (c.stance_duration + c.swing_duration)
                         for c in params.cycles]
        swing_ratios = [1.0 - s for s in stance_ratios]

        ax.bar(cycle_indices, stance_ratios, label="Stance", color="#2196F3", alpha=0.7)
        ax.bar(cycle_indices, swing_ratios, bottom=stance_ratios,
               label="Swing", color="#4CAF50", alpha=0.7)
        ax.axhline(0.6, color="gray", linestyle="--", alpha=0.5, label="Normal ~60%")
        ax.set_ylabel("Phase Ratio")
        ax.set_xlabel("Gait Cycle")
        ax.set_title(f"Stance/Swing Ratio per Cycle (n={len(params.cycles)}, "
                     f"cadence={params.mean_cadence:.1f} steps/min)")
        ax.legend(loc="upper right")
        plot_idx += 1

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
