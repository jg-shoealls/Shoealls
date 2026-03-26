"""Personal gait profile learner: builds and tracks individual baselines."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .foot_zones import FootZoneAnalyzer, ZONE_DEFINITIONS


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

    # Z-score thresholds for alerts
    MILD_THRESHOLD = 1.5
    MODERATE_THRESHOLD = 2.0
    SEVERE_THRESHOLD = 3.0

    def __init__(self, grid_h: int = 16, grid_w: int = 8):
        self.foot_analyzer = FootZoneAnalyzer(grid_h, grid_w)
        self.baseline: Optional[GaitBaseline] = None
        self._session_history: list[dict] = []

    def extract_session_features(
        self,
        pressure_seq: np.ndarray,
        imu_seq: Optional[np.ndarray] = None,
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

    def _compute_step_symmetry(self, accel_mag: np.ndarray) -> float:
        """Symmetry from autocorrelation: ratio of first two peaks."""
        if len(accel_mag) < 20:
            return 1.0
        ac = np.correlate(accel_mag - accel_mag.mean(), accel_mag - accel_mag.mean(), mode="full")
        ac = ac[len(ac) // 2:]
        ac = ac / (ac[0] + 1e-8)

        # Find first two peaks after lag 0
        peaks = []
        for i in range(2, len(ac) - 1):
            if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > 0.1:
                peaks.append((i, ac[i]))
            if len(peaks) >= 2:
                break

        if len(peaks) < 2:
            return 1.0
        return float(min(peaks[0][1], peaks[1][1]) / (max(peaks[0][1], peaks[1][1]) + 1e-8))

    def _estimate_cadence(self, accel_mag: np.ndarray, sample_rate: int = 128) -> float:
        """Estimate steps per minute from acceleration magnitude."""
        if len(accel_mag) < sample_rate:
            return 0.0
        # Simple peak counting
        mean_val = accel_mag.mean()
        above = accel_mag > mean_val
        crossings = np.diff(above.astype(int))
        num_steps = np.sum(crossings == 1)
        duration_sec = len(accel_mag) / sample_rate
        return float(num_steps / duration_sec * 60) if duration_sec > 0 else 0.0

    def _compute_stride_regularity(self, accel_mag: np.ndarray) -> float:
        """Stride regularity from autocorrelation peak height."""
        if len(accel_mag) < 20:
            return 1.0
        ac = np.correlate(accel_mag - accel_mag.mean(), accel_mag - accel_mag.mean(), mode="full")
        ac = ac[len(ac) // 2:]
        ac = ac / (ac[0] + 1e-8)

        # Find the dominant autocorrelation peak (stride period)
        best_peak = 0.0
        for i in range(5, len(ac) - 1):
            if ac[i] > ac[i - 1] and ac[i] > ac[i + 1]:
                best_peak = max(best_peak, ac[i])
        return float(min(best_peak, 1.0))

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
        scalar_metrics = {
            "ml_index": ("내외측 체중 분포", self.baseline.ml_index),
            "ap_index": ("전후방 체중 분포", self.baseline.ap_index),
            "arch_index": ("아치 지수", self.baseline.arch_index),
            "cop_sway": ("체중심 흔들림", self.baseline.cop_sway),
            "cadence": ("보행 속도(분당 걸음)", self.baseline.cadence),
            "stride_regularity": ("보폭 규칙성", self.baseline.stride_regularity),
            "step_symmetry": ("좌우 대칭성", self.baseline.step_symmetry),
            "acceleration_rms": ("가속도 크기", self.baseline.acceleration_rms),
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
