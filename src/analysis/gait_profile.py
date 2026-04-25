"""Personal gait profile learner: builds and tracks individual baselines."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .foot_zones import FootZoneAnalyzer, ZONE_DEFINITIONS
from .config import DEVIATION_THRESHOLDS
from .common import get_feature_korean


@dataclass
class GaitBaseline:
    """Statistical baseline for an individual's gait pattern."""
    # Pressure zone baselines: {zone_name: {"mean": float, "std": float, "peak_mean": float}}
    zone_baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    # Global gait indices
    ml_index: tuple[float, float] = (0.0, 0.0)  # (mean, std)
    ap_index: tuple[float, float] = (0.0, 0.0)
    arch_index: tuple[float, float] = (0.0, 0.0)
    cop_sway: tuple[float, float] = (0.0, 0.0)
    cadence: tuple[float, float] = (0.0, 0.0)  # steps/min
    stride_regularity: tuple[float, float] = (0.0, 0.0)
    # IMU-derived
    step_symmetry: tuple[float, float] = (0.0, 0.0)
    acceleration_rms: tuple[float, float] = (0.0, 0.0)
    # Number of sessions used to build baseline
    num_sessions: int = 0


@dataclass
class DeviationReport:
    """Report of how a session deviates from baseline."""
    deviations: dict[str, float]  # metric_name -> z-score
    alerts: list[dict[str, str]]  # list of {"metric", "severity", "message"}
    overall_deviation: float       # aggregate deviation score 0-1


class PersonalGaitProfiler:
    """Learns and tracks individual gait patterns over time.

    Maintains a running baseline from past sessions and flags deviations.
    """

    MILD_THRESHOLD = DEVIATION_THRESHOLDS["mild"]
    MODERATE_THRESHOLD = DEVIATION_THRESHOLDS["moderate"]
    SEVERE_THRESHOLD = DEVIATION_THRESHOLDS["severe"]

    def __init__(self, grid_h: int = 16, grid_w: int = 8):
        self.foot_analyzer = FootZoneAnalyzer(grid_h, grid_w)
        self.baseline: GaitBaseline | None = None
        self._session_history: list[dict] = []

    def extract_session_features(
        self,
        pressure_seq: np.ndarray,
        imu_seq: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Extract gait features from a single session.

        Args:
            pressure_seq: (T, 1, H, W) or (T, H, W) pressure data.
            imu_seq: Optional (C, T) IMU data (6-channel: accel xyz + gyro xyz).

        Returns:
            Dict of scalar feature values.
        """
        pa = self.foot_analyzer.analyze_sequence(pressure_seq)
        features = {}

        # Pressure zone features
        for zone_name in ZONE_DEFINITIONS:
            zt = pa["zone_temporal"][zone_name]
            features[f"zone_{zone_name}_mean"] = zt["mean_pressure_avg"]
            features[f"zone_{zone_name}_peak"] = zt["peak_pressure_max"]

        # Global pressure indices
        features["ml_index"] = pa["ml_index_mean"]
        features["ap_index"] = pa["ap_index_mean"]
        features["cop_sway"] = pa["cop_sway"]

        # Arch index (average across frames)
        arch_vals = [f.arch_index for f in pa["frames"]]
        features["arch_index"] = float(np.mean(arch_vals))

        # IMU-derived features
        if imu_seq is not None:
            imu = np.asarray(imu_seq, dtype=np.float64)
            if imu.ndim == 2 and imu.shape[0] <= 6:
                accel = imu[:3]  # (3, T)
            else:
                accel = imu[:, :3].T if imu.shape[1] >= 3 else imu[:3]

            accel_mag = np.sqrt(np.sum(accel ** 2, axis=0))
            features["acceleration_rms"] = float(np.sqrt(np.mean(accel_mag ** 2)))

            # Step symmetry from autocorrelation
            features["step_symmetry"] = self._compute_step_symmetry(accel_mag)

            # Cadence estimation from peak detection
            features["cadence"] = self._estimate_cadence(accel_mag, sample_rate=128)

            # Stride regularity from autocorrelation
            features["stride_regularity"] = self._compute_stride_regularity(accel_mag)

        return features

    @staticmethod
    def _autocorr(accel_mag: np.ndarray) -> np.ndarray:
        """정규화된 자동상관 계수 반환 (lag=0 이후)."""
        centered = accel_mag - accel_mag.mean()
        ac = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
        return ac / (ac[0] + 1e-8)

    def _compute_step_symmetry(self, accel_mag: np.ndarray) -> float:
        """Symmetry from autocorrelation: ratio of first two peaks."""
        if len(accel_mag) < 20:
            return 1.0
        ac = self._autocorr(accel_mag)
        # 연속 피크 2개 탐색 (lag ≥ 2, threshold > 0.1)
        peaks = [(i, ac[i]) for i in range(2, len(ac) - 1)
                 if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > 0.1][:2]
        if len(peaks) < 2:
            return 1.0
        v0, v1 = peaks[0][1], peaks[1][1]
        return float(min(v0, v1) / (max(v0, v1) + 1e-8))

    def _estimate_cadence(self, accel_mag: np.ndarray, sample_rate: int = 128) -> float:
        """Estimate steps per minute from acceleration magnitude."""
        if len(accel_mag) < sample_rate:
            return 0.0
        above = accel_mag > accel_mag.mean()
        num_steps = int(np.sum(np.diff(above.astype(np.int8)) == 1))
        duration_sec = len(accel_mag) / sample_rate
        return float(num_steps / duration_sec * 60) if duration_sec > 0 else 0.0

    def _compute_stride_regularity(self, accel_mag: np.ndarray) -> float:
        """Stride regularity from autocorrelation peak height."""
        if len(accel_mag) < 20:
            return 1.0
        ac = self._autocorr(accel_mag)
        # lag ≥ 5 에서 전체 최대 피크
        inner = ac[5:-1]
        if len(inner) == 0:
            return 0.0
        local_max = inner[(inner > np.roll(inner, 1)) & (inner > np.roll(inner, -1))]
        return float(min(local_max.max(), 1.0)) if len(local_max) > 0 else 0.0

    def update_baseline(self, session_features: dict[str, float]):
        """Update the running baseline with new session features.

        Uses incremental mean/variance (Welford's algorithm).
        """
        self._session_history.append(session_features)

        if self.baseline is None:
            self.baseline = GaitBaseline()

        n = len(self._session_history)
        self.baseline.num_sessions = n

        if n == 1:
            # First session: initialize
            for key, val in session_features.items():
                if key.startswith("zone_"):
                    zone = key.replace("zone_", "").rsplit("_", 1)[0]
                    suffix = key.rsplit("_", 1)[-1]
                    if zone not in self.baseline.zone_baselines:
                        self.baseline.zone_baselines[zone] = {}
                    self.baseline.zone_baselines[zone][suffix] = val
                    self.baseline.zone_baselines[zone][f"{suffix}_std"] = 0.0
            self.baseline.ml_index = (session_features.get("ml_index", 0.0), 0.0)
            self.baseline.ap_index = (session_features.get("ap_index", 0.0), 0.0)
            self.baseline.arch_index = (session_features.get("arch_index", 0.0), 0.0)
            self.baseline.cop_sway = (session_features.get("cop_sway", 0.0), 0.0)
            self.baseline.cadence = (session_features.get("cadence", 0.0), 0.0)
            self.baseline.stride_regularity = (session_features.get("stride_regularity", 0.0), 0.0)
            self.baseline.step_symmetry = (session_features.get("step_symmetry", 0.0), 0.0)
            self.baseline.acceleration_rms = (session_features.get("acceleration_rms", 0.0), 0.0)
        else:
            # Recompute from full history
            all_features = {}
            for sf in self._session_history:
                for key, val in sf.items():
                    all_features.setdefault(key, []).append(val)

            # Update zone baselines
            for key, vals in all_features.items():
                if key.startswith("zone_"):
                    zone = key.replace("zone_", "").rsplit("_", 1)[0]
                    suffix = key.rsplit("_", 1)[-1]
                    if zone not in self.baseline.zone_baselines:
                        self.baseline.zone_baselines[zone] = {}
                    self.baseline.zone_baselines[zone][suffix] = float(np.mean(vals))
                    self.baseline.zone_baselines[zone][f"{suffix}_std"] = float(np.std(vals))

            for attr in ["ml_index", "ap_index", "arch_index", "cop_sway",
                         "cadence", "stride_regularity", "step_symmetry", "acceleration_rms"]:
                if attr in all_features:
                    vals = all_features[attr]
                    setattr(self.baseline, attr, (float(np.mean(vals)), float(np.std(vals))))

    def compute_deviations(self, session_features: dict[str, float]) -> DeviationReport:
        """Compare session features against baseline and report deviations."""
        if self.baseline is None or self.baseline.num_sessions < 2:
            return DeviationReport(deviations={}, alerts=[], overall_deviation=0.0)

        deviations = {}
        alerts = []

        # Check scalar metrics
        _baseline_map = {
            "ml_index": self.baseline.ml_index,
            "ap_index": self.baseline.ap_index,
            "arch_index": self.baseline.arch_index,
            "cop_sway": self.baseline.cop_sway,
            "cadence": self.baseline.cadence,
            "stride_regularity": self.baseline.stride_regularity,
            "step_symmetry": self.baseline.step_symmetry,
            "acceleration_rms": self.baseline.acceleration_rms,
        }
        scalar_metrics = {
            k: (get_feature_korean(k), v) for k, v in _baseline_map.items()
        }

        for metric, (korean_name, (mean, std)) in scalar_metrics.items():
            if metric not in session_features:
                continue
            val = session_features[metric]
            z = abs(val - mean) / (std + 1e-8) if std > 1e-8 else 0.0
            deviations[metric] = z

            if z >= self.SEVERE_THRESHOLD:
                alerts.append({
                    "metric": metric,
                    "severity": "심각",
                    "message": f"{korean_name}이(가) 평소 대비 크게 벗어났습니다 (z={z:.1f})",
                })
            elif z >= self.MODERATE_THRESHOLD:
                alerts.append({
                    "metric": metric,
                    "severity": "주의",
                    "message": f"{korean_name}이(가) 평소 대비 변화가 감지되었습니다 (z={z:.1f})",
                })
            elif z >= self.MILD_THRESHOLD:
                alerts.append({
                    "metric": metric,
                    "severity": "경미",
                    "message": f"{korean_name}에 약간의 변화가 있습니다 (z={z:.1f})",
                })

        # Overall deviation: RMS of z-scores
        if deviations:
            z_vals = list(deviations.values())
            overall = float(np.sqrt(np.mean(np.array(z_vals) ** 2)))
            # Normalize to 0-1 range (3+ sigma → 1.0)
            overall = min(overall / 3.0, 1.0)
        else:
            overall = 0.0

        return DeviationReport(
            deviations=deviations,
            alerts=sorted(alerts, key=lambda a: {"심각": 0, "주의": 1, "경미": 2}[a["severity"]]),
            overall_deviation=overall,
        )
