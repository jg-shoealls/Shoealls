"""압력 센서 비선형성 보정 및 실시간 체중 측정 모듈.

Pressure sensor nonlinearity correction and real-time weight estimation.

FSR(Force Sensing Resistor) 기반 압력 센서는 비선형 전압-힘 특성을 가집니다.
이 모듈은 다항식/스플라인 보정, 셀별 개별 캘리브레이션, 온도 보상,
드리프트 보정을 통해 정확한 체중 측정을 수행합니다.

주요 기능:
    1. 센서별 비선형 응답 보정 (다항식, 스플라인, 구간선형)
    2. 실시간 체중 추정 (전체 압력 합산 → 힘 → 체중)
    3. 온도 보상 (센서 감도 온도 의존성 보정)
    4. 드리프트 보정 (장시간 사용 시 기준선 이동 보정)
    5. 캘리브레이션 프로파일 저장/로드
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationProfile:
    """센서 캘리브레이션 프로파일.

    각 셀(또는 전체 센서)의 보정 파라미터를 저장합니다.

    Attributes:
        sensor_id: 센서 식별자.
        grid_size: 압력 그리드 크기 (H, W).
        method: 보정 방법 ('polynomial', 'spline', 'piecewise').
        poly_degree: 다항식 차수 (polynomial 방식일 때).
        coefficients: 셀별 보정 계수. shape (H, W, degree+1) 또는 None.
        reference_points: 캘리브레이션 기준점 (raw, true_force) 쌍.
        temperature_coeff: 온도 보상 계수 (% / °C).
        reference_temperature: 기준 온도 (°C).
        zero_offset: 셀별 영점 오프셋. shape (H, W).
        created_at: 프로파일 생성 시각 (ISO 8601).
    """

    sensor_id: str = "default"
    grid_size: tuple[int, int] = (16, 8)
    method: str = "polynomial"
    poly_degree: int = 3
    coefficients: Optional[np.ndarray] = None
    reference_points: list[tuple[float, float]] = field(default_factory=list)
    temperature_coeff: float = -0.3  # FSR 일반적 값: -0.3%/°C
    reference_temperature: float = 25.0
    zero_offset: Optional[np.ndarray] = None
    created_at: str = ""

    def save(self, path: str) -> None:
        """프로파일을 JSON 파일로 저장."""
        data = {
            "sensor_id": self.sensor_id,
            "grid_size": list(self.grid_size),
            "method": self.method,
            "poly_degree": self.poly_degree,
            "reference_points": self.reference_points,
            "temperature_coeff": self.temperature_coeff,
            "reference_temperature": self.reference_temperature,
            "created_at": self.created_at,
        }
        if self.coefficients is not None:
            data["coefficients"] = self.coefficients.tolist()
        if self.zero_offset is not None:
            data["zero_offset"] = self.zero_offset.tolist()

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Calibration profile saved: %s", path)

    @classmethod
    def load(cls, path: str) -> "CalibrationProfile":
        """JSON 파일에서 프로파일 로드."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile = cls(
            sensor_id=data["sensor_id"],
            grid_size=tuple(data["grid_size"]),
            method=data["method"],
            poly_degree=data["poly_degree"],
            reference_points=data.get("reference_points", []),
            temperature_coeff=data.get("temperature_coeff", -0.3),
            reference_temperature=data.get("reference_temperature", 25.0),
            created_at=data.get("created_at", ""),
        )
        if "coefficients" in data:
            profile.coefficients = np.array(data["coefficients"])
        if "zero_offset" in data:
            profile.zero_offset = np.array(data["zero_offset"])
        return profile


@dataclass
class WeightEstimate:
    """체중 측정 결과.

    Attributes:
        weight_kg: 추정 체중 (kg).
        total_force_n: 총 힘 (N).
        confidence: 추정 신뢰도 (0~1).
        left_ratio: 좌측 체중 부하 비율 (0~1).
        right_ratio: 우측 체중 부하 비율 (0~1).
        cop_x: 압력 중심 X (정규화 0~1).
        cop_y: 압력 중심 Y (정규화 0~1).
        is_stable: 안정 상태 여부 (정지 시 True).
    """

    weight_kg: float = 0.0
    total_force_n: float = 0.0
    confidence: float = 0.0
    left_ratio: float = 0.5
    right_ratio: float = 0.5
    cop_x: float = 0.5
    cop_y: float = 0.5
    is_stable: bool = False


# ---------------------------------------------------------------------------
# Nonlinearity correction
# ---------------------------------------------------------------------------

class PressureCalibrator:
    """압력 센서 비선형성 보정기.

    FSR 센서의 비선형 전압-힘 응답을 보정합니다.
    셀별 개별 캘리브레이션 또는 전역 캘리브레이션을 지원합니다.

    보정 방법:
        - polynomial: N차 다항식 피팅 (기본 3차)
        - spline: 큐빅 스플라인 보간
        - piecewise: 구간별 선형 보간

    Example::

        calibrator = PressureCalibrator(grid_size=(16, 8))

        # 캘리브레이션 데이터로 보정 함수 학습
        raw_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        true_forces = [0.0, 2.5, 8.0, 18.0, 35.0, 50.0]  # Newton
        calibrator.fit(raw_values, true_forces)

        # 실시간 보정 적용
        corrected = calibrator.correct(raw_pressure_frame)
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (16, 8),
        method: str = "polynomial",
        poly_degree: int = 3,
    ):
        self.grid_size = grid_size
        self.method = method
        self.poly_degree = poly_degree

        self._global_coeffs: Optional[np.ndarray] = None
        self._cell_coeffs: Optional[np.ndarray] = None  # (H, W, degree+1)
        self._spline_fn: Optional[interp1d] = None
        self._cell_spline_fns: Optional[list] = None
        self._piecewise_fn: Optional[interp1d] = None
        self._profile: Optional[CalibrationProfile] = None

    def fit(
        self,
        raw_values: list[float] | np.ndarray,
        true_forces: list[float] | np.ndarray,
    ) -> None:
        """전역 캘리브레이션: 모든 셀에 동일한 보정 곡선 적용.

        Args:
            raw_values: 센서 원시 판독값 (ADC 또는 정규화 0~1).
            true_forces: 대응하는 실제 힘 값 (Newton).
        """
        raw = np.asarray(raw_values, dtype=np.float64)
        force = np.asarray(true_forces, dtype=np.float64)

        if len(raw) < 2:
            raise ValueError("At least 2 calibration points required")
        if len(raw) != len(force):
            raise ValueError("raw_values and true_forces must have same length")

        # 정렬
        sort_idx = np.argsort(raw)
        raw = raw[sort_idx]
        force = force[sort_idx]

        if self.method == "polynomial":
            self._global_coeffs = np.polyfit(raw, force, self.poly_degree)
            logger.info(
                "Polynomial fit (degree %d): coeffs=%s",
                self.poly_degree,
                np.round(self._global_coeffs, 4),
            )

        elif self.method == "spline":
            self._spline_fn = interp1d(
                raw, force, kind="cubic", fill_value="extrapolate",
            )
            logger.info("Spline fit with %d points", len(raw))

        elif self.method == "piecewise":
            self._piecewise_fn = interp1d(
                raw, force, kind="linear", fill_value="extrapolate",
            )
            logger.info("Piecewise linear fit with %d segments", len(raw) - 1)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # 프로파일 업데이트
        self._profile = CalibrationProfile(
            grid_size=self.grid_size,
            method=self.method,
            poly_degree=self.poly_degree,
            reference_points=list(zip(raw.tolist(), force.tolist())),
        )
        if self._global_coeffs is not None:
            self._profile.coefficients = self._global_coeffs

    def fit_per_cell(
        self,
        raw_maps: np.ndarray,
        true_force_maps: np.ndarray,
    ) -> None:
        """셀별 개별 캘리브레이션.

        각 셀(h, w)에 대해 독립적인 보정 곡선을 학습합니다.

        Args:
            raw_maps: (N, H, W) N개 캘리브레이션 포인트의 원시 압력 맵.
            true_force_maps: (N, H, W) 대응하는 실제 힘 맵.
        """
        N, H, W = raw_maps.shape
        if (H, W) != self.grid_size:
            raise ValueError(f"Grid size mismatch: expected {self.grid_size}, got ({H}, {W})")

        self._cell_coeffs = np.zeros((H, W, self.poly_degree + 1))
        self._cell_spline_fns = []

        for h in range(H):
            row_fns = []
            for w in range(W):
                raw_cell = raw_maps[:, h, w]
                force_cell = true_force_maps[:, h, w]

                sort_idx = np.argsort(raw_cell)
                raw_sorted = raw_cell[sort_idx]
                force_sorted = force_cell[sort_idx]

                if self.method == "polynomial":
                    coeffs = np.polyfit(raw_sorted, force_sorted, self.poly_degree)
                    self._cell_coeffs[h, w, :] = coeffs
                    row_fns.append(None)
                elif self.method == "spline":
                    fn = interp1d(
                        raw_sorted, force_sorted,
                        kind="cubic", fill_value="extrapolate",
                    )
                    row_fns.append(fn)
                else:
                    fn = interp1d(
                        raw_sorted, force_sorted,
                        kind="linear", fill_value="extrapolate",
                    )
                    row_fns.append(fn)

            self._cell_spline_fns.append(row_fns)

        self._profile = CalibrationProfile(
            grid_size=self.grid_size,
            method=self.method,
            poly_degree=self.poly_degree,
            coefficients=self._cell_coeffs,
        )
        logger.info("Per-cell calibration complete: %dx%d cells", H, W)

    def correct(self, raw_frame: np.ndarray) -> np.ndarray:
        """단일 프레임의 압력 값을 보정된 힘(N)으로 변환.

        Args:
            raw_frame: (H, W) 원시 압력 프레임.

        Returns:
            (H, W) 보정된 힘 맵 (Newton).
        """
        if raw_frame.ndim != 2:
            raise ValueError(f"Expected (H, W) array, got shape {raw_frame.shape}")

        # 셀별 보정이 있으면 사용
        if self._cell_coeffs is not None:
            return self._correct_per_cell(raw_frame)

        # 전역 보정
        if self.method == "polynomial" and self._global_coeffs is not None:
            corrected = np.polyval(self._global_coeffs, raw_frame)
        elif self.method == "spline" and self._spline_fn is not None:
            corrected = self._spline_fn(raw_frame)
        elif self.method == "piecewise" and self._piecewise_fn is not None:
            corrected = self._piecewise_fn(raw_frame)
        else:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        return np.maximum(corrected, 0.0).astype(np.float32)

    def correct_sequence(self, raw_sequence: np.ndarray) -> np.ndarray:
        """시계열 압력 데이터 전체 보정.

        Args:
            raw_sequence: (T, H, W) 원시 압력 시계열.

        Returns:
            (T, H, W) 보정된 힘 시계열 (Newton).
        """
        if raw_sequence.ndim != 3:
            raise ValueError(f"Expected (T, H, W), got shape {raw_sequence.shape}")
        return np.stack([self.correct(frame) for frame in raw_sequence])

    def _correct_per_cell(self, raw_frame: np.ndarray) -> np.ndarray:
        """셀별 개별 보정."""
        H, W = raw_frame.shape
        corrected = np.zeros_like(raw_frame, dtype=np.float32)

        for h in range(H):
            for w in range(W):
                val = raw_frame[h, w]
                if self.method == "polynomial":
                    corrected[h, w] = np.polyval(self._cell_coeffs[h, w], val)
                elif self._cell_spline_fns is not None:
                    corrected[h, w] = float(self._cell_spline_fns[h][w](val))

        return np.maximum(corrected, 0.0)

    @property
    def profile(self) -> Optional[CalibrationProfile]:
        return self._profile


# ---------------------------------------------------------------------------
# Temperature compensation
# ---------------------------------------------------------------------------

class TemperatureCompensator:
    """온도 보상기.

    FSR 센서는 온도에 따라 감도가 변합니다 (일반적으로 -0.1~-0.5%/°C).
    기준 온도 대비 현재 온도 차이를 보정합니다.

    Args:
        temp_coeff: 온도 계수 (%/°C). 음수면 온도 상승 시 감도 감소.
        reference_temp: 캘리브레이션 기준 온도 (°C).
    """

    def __init__(self, temp_coeff: float = -0.3, reference_temp: float = 25.0):
        self.temp_coeff = temp_coeff
        self.reference_temp = reference_temp

    def compensate(
        self,
        force_map: np.ndarray,
        current_temp: float,
    ) -> np.ndarray:
        """온도 보상 적용.

        Args:
            force_map: (H, W) 또는 (T, H, W) 보정된 힘 맵.
            current_temp: 현재 센서 온도 (°C).

        Returns:
            온도 보상된 힘 맵.
        """
        delta_t = current_temp - self.reference_temp
        # 보상 계수: 온도가 높으면 센서 출력이 낮아지므로 보정값을 올림
        compensation_factor = 1.0 / (1.0 + self.temp_coeff / 100.0 * delta_t)
        return (force_map * compensation_factor).astype(np.float32)


# ---------------------------------------------------------------------------
# Drift correction
# ---------------------------------------------------------------------------

class DriftCorrector:
    """기준선 드리프트 보정기.

    장시간 사용 시 FSR 센서의 기준선(zero-load output)이 서서히 변하는
    현상을 실시간으로 보정합니다.

    적응형 기준선 추정:
        - 비접지 구간(unloaded)을 자동 감지
        - 지수 이동 평균(EMA)으로 기준선 업데이트
        - 보정 = 원시값 - 추정된 기준선

    Args:
        grid_size: 압력 그리드 크기 (H, W).
        ema_alpha: 기준선 업데이트 속도 (0~1). 작을수록 느리게 적응.
        unloaded_threshold: 비접지 판정 임계값 (원시값 기준).
        initial_baseline: 초기 기준선 값.
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (16, 8),
        ema_alpha: float = 0.01,
        unloaded_threshold: float = 0.05,
        initial_baseline: float = 0.0,
    ):
        H, W = grid_size
        self.ema_alpha = ema_alpha
        self.unloaded_threshold = unloaded_threshold
        self.baseline = np.full((H, W), initial_baseline, dtype=np.float32)
        self._frame_count = 0

    def update_and_correct(self, raw_frame: np.ndarray) -> np.ndarray:
        """기준선 업데이트 후 드리프트 보정 적용.

        Args:
            raw_frame: (H, W) 원시 압력 프레임.

        Returns:
            (H, W) 드리프트 보정된 프레임.
        """
        self._frame_count += 1

        # 비접지 셀 마스크 (아주 낮은 압력 = 발이 안 닿은 부분)
        unloaded_mask = raw_frame < self.unloaded_threshold

        # 비접지 셀의 기준선만 EMA 업데이트
        if unloaded_mask.any():
            self.baseline[unloaded_mask] = (
                (1 - self.ema_alpha) * self.baseline[unloaded_mask]
                + self.ema_alpha * raw_frame[unloaded_mask]
            )

        # 드리프트 보정
        corrected = raw_frame - self.baseline
        return np.maximum(corrected, 0.0).astype(np.float32)

    def reset(self) -> None:
        """기준선 초기화."""
        self.baseline[:] = 0.0
        self._frame_count = 0


# ---------------------------------------------------------------------------
# Real-time weight estimation
# ---------------------------------------------------------------------------

class RealtimeWeightEstimator:
    """실시간 체중 측정기.

    보정된 압력 맵에서 총 힘을 계산하고 체중으로 변환합니다.
    안정성 판단, 좌우 비율, 압력 중심(CoP)을 실시간으로 출력합니다.

    파이프라인:
        raw pressure → drift correction → nonlinearity correction
        → temperature compensation → force summation → weight (kg)

    Args:
        calibrator: 비선형성 보정기.
        drift_corrector: 드리프트 보정기 (선택).
        temp_compensator: 온도 보상기 (선택).
        sensor_area_cm2: 셀당 면적 (cm²). 기본값은 16x8 그리드 기준.
        stability_window: 안정성 판단 윈도우 (프레임 수).
        stability_threshold: 체중 변동 임계값 (kg). 이하면 안정.
    """

    GRAVITY = 9.80665  # m/s²

    def __init__(
        self,
        calibrator: PressureCalibrator,
        drift_corrector: Optional[DriftCorrector] = None,
        temp_compensator: Optional[TemperatureCompensator] = None,
        sensor_area_cm2: float = 0.5,
        stability_window: int = 30,
        stability_threshold: float = 0.5,
    ):
        self.calibrator = calibrator
        self.drift_corrector = drift_corrector
        self.temp_compensator = temp_compensator
        self.sensor_area_cm2 = sensor_area_cm2
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold

        self._weight_history: list[float] = []
        self._grid_size = calibrator.grid_size

    def estimate(
        self,
        raw_frame: np.ndarray,
        temperature: Optional[float] = None,
    ) -> WeightEstimate:
        """단일 프레임에서 체중 추정.

        Args:
            raw_frame: (H, W) 원시 압력 프레임.
            temperature: 현재 센서 온도 (°C). None이면 온도 보상 생략.

        Returns:
            WeightEstimate 결과.
        """
        frame = raw_frame.astype(np.float32)

        # 1. 드리프트 보정
        if self.drift_corrector is not None:
            frame = self.drift_corrector.update_and_correct(frame)

        # 2. 비선형성 보정 (raw → Newton)
        force_map = self.calibrator.correct(frame)

        # 3. 온도 보상
        if self.temp_compensator is not None and temperature is not None:
            force_map = self.temp_compensator.compensate(force_map, temperature)

        # 4. 총 힘 계산
        total_force_n = float(force_map.sum())

        # 5. 체중 변환 (N → kg)
        weight_kg = total_force_n / self.GRAVITY

        # 6. 좌우 비율 (좌: cols 0~W//2, 우: cols W//2~W)
        H, W = force_map.shape
        left_force = force_map[:, : W // 2].sum()
        right_force = force_map[:, W // 2 :].sum()
        total_lr = left_force + right_force + 1e-8
        left_ratio = float(left_force / total_lr)
        right_ratio = float(right_force / total_lr)

        # 7. 압력 중심 (CoP)
        cop_y, cop_x = self._compute_cop(force_map)

        # 8. 안정성 판단
        self._weight_history.append(weight_kg)
        if len(self._weight_history) > self.stability_window:
            self._weight_history = self._weight_history[-self.stability_window :]
        is_stable = self._check_stability()

        # 9. 신뢰도 계산
        confidence = self._compute_confidence(force_map, is_stable)

        return WeightEstimate(
            weight_kg=round(weight_kg, 2),
            total_force_n=round(total_force_n, 2),
            confidence=round(confidence, 3),
            left_ratio=round(left_ratio, 3),
            right_ratio=round(right_ratio, 3),
            cop_x=round(cop_x, 4),
            cop_y=round(cop_y, 4),
            is_stable=is_stable,
        )

    def estimate_sequence(
        self,
        raw_sequence: np.ndarray,
        temperature: Optional[float] = None,
    ) -> list[WeightEstimate]:
        """시계열 압력 데이터에서 프레임별 체중 추정.

        Args:
            raw_sequence: (T, H, W) 원시 압력 시계열.
            temperature: 센서 온도 (°C).

        Returns:
            프레임별 WeightEstimate 리스트.
        """
        return [self.estimate(frame, temperature) for frame in raw_sequence]

    def get_stable_weight(
        self,
        raw_sequence: np.ndarray,
        temperature: Optional[float] = None,
        min_stable_frames: int = 20,
    ) -> Optional[WeightEstimate]:
        """안정 구간의 평균 체중 반환.

        연속 안정 프레임이 min_stable_frames 이상인 구간의
        평균 체중을 반환합니다.

        Args:
            raw_sequence: (T, H, W) 원시 압력 시계열.
            temperature: 센서 온도 (°C).
            min_stable_frames: 최소 안정 프레임 수.

        Returns:
            안정 구간 평균 WeightEstimate 또는 None.
        """
        estimates = self.estimate_sequence(raw_sequence, temperature)

        # 안정 구간 탐색
        best_run: list[WeightEstimate] = []
        current_run: list[WeightEstimate] = []

        for est in estimates:
            if est.is_stable:
                current_run.append(est)
            else:
                if len(current_run) > len(best_run):
                    best_run = current_run
                current_run = []

        if len(current_run) > len(best_run):
            best_run = current_run

        if len(best_run) < min_stable_frames:
            return None

        # 안정 구간 평균
        weights = [e.weight_kg for e in best_run]
        forces = [e.total_force_n for e in best_run]
        return WeightEstimate(
            weight_kg=round(float(np.mean(weights)), 2),
            total_force_n=round(float(np.mean(forces)), 2),
            confidence=round(float(np.mean([e.confidence for e in best_run])), 3),
            left_ratio=round(float(np.mean([e.left_ratio for e in best_run])), 3),
            right_ratio=round(float(np.mean([e.right_ratio for e in best_run])), 3),
            cop_x=round(float(np.mean([e.cop_x for e in best_run])), 4),
            cop_y=round(float(np.mean([e.cop_y for e in best_run])), 4),
            is_stable=True,
        )

    def _compute_cop(self, force_map: np.ndarray) -> tuple[float, float]:
        """압력 중심(Center of Pressure) 계산.

        Returns:
            (cop_y, cop_x) 정규화된 좌표 (0~1).
        """
        H, W = force_map.shape
        total = force_map.sum()
        if total < 1e-6:
            return 0.5, 0.5

        rows = np.arange(H).reshape(-1, 1)
        cols = np.arange(W).reshape(1, -1)
        cop_y = float((force_map * rows).sum() / total) / (H - 1)
        cop_x = float((force_map * cols).sum() / total) / (W - 1)
        return np.clip(cop_y, 0, 1), np.clip(cop_x, 0, 1)

    def _check_stability(self) -> bool:
        """최근 윈도우의 체중 변동으로 안정성 판단."""
        if len(self._weight_history) < self.stability_window:
            return False
        recent = self._weight_history[-self.stability_window :]
        return float(np.std(recent)) < self.stability_threshold

    def _compute_confidence(
        self,
        force_map: np.ndarray,
        is_stable: bool,
    ) -> float:
        """측정 신뢰도 계산.

        고려 요소:
            - 전체 힘의 크기 (너무 작으면 노이즈 지배적)
            - 활성 셀 비율 (접촉 면적이 합리적인 범위인지)
            - 안정성 (변동이 적을수록 높은 신뢰도)
        """
        H, W = force_map.shape
        total_cells = H * W

        # 활성 셀 비율 (0.1N 이상)
        active_ratio = (force_map > 0.1).sum() / total_cells

        # 활성 비율 점수 (10~60%가 적정 범위)
        if 0.1 <= active_ratio <= 0.6:
            area_score = 1.0
        elif active_ratio < 0.1:
            area_score = active_ratio / 0.1
        else:
            area_score = max(0.5, 1.0 - (active_ratio - 0.6) / 0.4)

        # 힘 크기 점수 (최소 100N = ~10kg 이상이면 OK)
        total_force = force_map.sum()
        force_score = min(total_force / 100.0, 1.0)

        # 안정성 점수
        stability_score = 0.9 if is_stable else 0.5

        return float(np.clip(
            0.4 * area_score + 0.3 * force_score + 0.3 * stability_score,
            0.0, 1.0,
        ))

    def reset(self) -> None:
        """상태 초기화."""
        self._weight_history.clear()
        if self.drift_corrector is not None:
            self.drift_corrector.reset()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_weight_estimator(
    calibration_profile_path: Optional[str] = None,
    raw_values: Optional[list[float]] = None,
    true_forces: Optional[list[float]] = None,
    grid_size: tuple[int, int] = (16, 8),
    method: str = "polynomial",
    enable_drift_correction: bool = True,
    enable_temp_compensation: bool = True,
) -> RealtimeWeightEstimator:
    """체중 측정기를 간편하게 생성하는 팩토리 함수.

    Args:
        calibration_profile_path: 기존 캘리브레이션 프로파일 JSON 경로.
        raw_values: 캘리브레이션 원시값 리스트 (프로파일 없을 때).
        true_forces: 캘리브레이션 실제 힘 리스트 (Newton).
        grid_size: 압력 그리드 크기.
        method: 보정 방법.
        enable_drift_correction: 드리프트 보정 활성화.
        enable_temp_compensation: 온도 보상 활성화.

    Returns:
        구성된 RealtimeWeightEstimator.
    """
    calibrator = PressureCalibrator(grid_size=grid_size, method=method)

    if calibration_profile_path and Path(calibration_profile_path).exists():
        profile = CalibrationProfile.load(calibration_profile_path)
        calibrator.grid_size = profile.grid_size
        calibrator.method = profile.method
        calibrator.poly_degree = profile.poly_degree
        if profile.coefficients is not None:
            if profile.coefficients.ndim == 1:
                calibrator._global_coeffs = profile.coefficients
            else:
                calibrator._cell_coeffs = profile.coefficients
        calibrator._profile = profile
    elif raw_values is not None and true_forces is not None:
        calibrator.fit(raw_values, true_forces)
    else:
        # FSR 일반적 비선형 응답 기본 캘리브레이션
        default_raw = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        default_force = [0.0, 0.5, 2.0, 5.0, 10.0, 17.0, 26.0, 37.0, 50.0, 65.0, 80.0]
        calibrator.fit(default_raw, default_force)
        logger.warning("Using default FSR calibration curve. Calibrate for accurate results.")

    drift = DriftCorrector(grid_size=grid_size) if enable_drift_correction else None
    temp = TemperatureCompensator() if enable_temp_compensation else None

    return RealtimeWeightEstimator(
        calibrator=calibrator,
        drift_corrector=drift,
        temp_compensator=temp,
    )
