"""Foot zone analyzer: maps pressure grid to anatomical regions and computes metrics."""

import numpy as np
from dataclasses import dataclass


# Anatomical zone definitions for a 16x8 pressure grid
# Row 0 = toe end, Row 15 = heel end
ZONE_DEFINITIONS = {
    "toes": {"rows": (0, 3), "cols": (0, 8), "description": "발가락 영역"},
    "forefoot_medial": {"rows": (3, 7), "cols": (0, 4), "description": "앞발 내측"},
    "forefoot_lateral": {"rows": (3, 7), "cols": (4, 8), "description": "앞발 외측"},
    "midfoot_medial": {"rows": (7, 11), "cols": (0, 4), "description": "중족부 내측"},
    "midfoot_lateral": {"rows": (7, 11), "cols": (4, 8), "description": "중족부 외측"},
    "heel_medial": {"rows": (11, 16), "cols": (0, 4), "description": "뒤꿈치 내측"},
    "heel_lateral": {"rows": (11, 16), "cols": (4, 8), "description": "뒤꿈치 외측"},
}

# Higher-level groupings
REGION_GROUPS = {
    "forefoot": ["toes", "forefoot_medial", "forefoot_lateral"],
    "midfoot": ["midfoot_medial", "midfoot_lateral"],
    "heel": ["heel_medial", "heel_lateral"],
    "medial": ["forefoot_medial", "midfoot_medial", "heel_medial"],
    "lateral": ["forefoot_lateral", "midfoot_lateral", "heel_lateral"],
}


@dataclass
class ZoneMetrics:
    """Pressure metrics for a single anatomical zone."""
    mean_pressure: float
    peak_pressure: float
    contact_area_ratio: float  # fraction of zone with nonzero pressure
    pressure_integral: float   # sum of pressures (force proxy)
    std_pressure: float


@dataclass
class FootAnalysisResult:
    """Complete foot pressure analysis for one frame or averaged over time."""
    zone_metrics: dict[str, ZoneMetrics]
    cop_x: float  # center of pressure, normalized 0-1
    cop_y: float
    mediolateral_index: float   # >0 = lateral shift, <0 = medial shift
    anteroposterior_index: float  # >0 = forefoot, <0 = heel
    arch_index: float           # midfoot / total contact ratio
    total_pressure: float
    pressure_symmetry: float    # left-right symmetry (0=perfect)


class FootZoneAnalyzer:
    """Analyzes plantar pressure distribution by anatomical zones.

    Operates on pressure grids of shape (T, 1, H, W) or (H, W).
    """

    def __init__(self, grid_h: int = 16, grid_w: int = 8, threshold: float = 0.05):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.threshold = threshold
        self._build_zone_masks()

    def _build_zone_masks(self):
        """Precompute binary masks for each zone."""
        self.zone_masks = {}
        for name, zdef in ZONE_DEFINITIONS.items():
            mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)
            r0, r1 = zdef["rows"]
            c0, c1 = zdef["cols"]
            mask[r0:r1, c0:c1] = True
            self.zone_masks[name] = mask

    def _compute_zone_metrics(self, pressure: np.ndarray, mask: np.ndarray) -> ZoneMetrics:
        """Compute metrics for one zone from a 2D pressure map."""
        zone_vals = pressure[mask]
        if zone_vals.size == 0:
            return ZoneMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        active = zone_vals > self.threshold
        return ZoneMetrics(
            mean_pressure=float(np.mean(zone_vals)),
            peak_pressure=float(np.max(zone_vals)),
            contact_area_ratio=float(np.sum(active) / zone_vals.size),
            pressure_integral=float(np.sum(zone_vals)),
            std_pressure=float(np.std(zone_vals)),
        )

    def _compute_cop(self, pressure: np.ndarray) -> tuple[float, float]:
        """Compute center of pressure (normalized 0-1)."""
        total = pressure.sum()
        if total < 1e-8:
            return 0.5, 0.5
        rows, cols = np.meshgrid(
            np.arange(pressure.shape[0]), np.arange(pressure.shape[1]), indexing="ij"
        )
        cop_y = float(np.sum(rows * pressure) / total) / max(pressure.shape[0] - 1, 1)
        cop_x = float(np.sum(cols * pressure) / total) / max(pressure.shape[1] - 1, 1)
        return cop_x, cop_y

    def analyze_frame(self, pressure_2d: np.ndarray) -> FootAnalysisResult:
        """Analyze a single pressure frame of shape (H, W)."""
        p = np.asarray(pressure_2d, dtype=np.float64)
        if p.ndim == 3 and p.shape[0] == 1:
            p = p[0]

        # Per-zone metrics
        zone_metrics = {}
        for name, mask in self.zone_masks.items():
            zone_metrics[name] = self._compute_zone_metrics(p, mask)

        # Center of pressure
        cop_x, cop_y = self._compute_cop(p)

        # Mediolateral index: lateral - medial pressure ratio
        medial_p = sum(zone_metrics[z].pressure_integral for z in REGION_GROUPS["medial"])
        lateral_p = sum(zone_metrics[z].pressure_integral for z in REGION_GROUPS["lateral"])
        total_ml = medial_p + lateral_p
        ml_index = (lateral_p - medial_p) / total_ml if total_ml > 1e-8 else 0.0

        # Anteroposterior index: forefoot - heel pressure ratio
        fore_p = sum(zone_metrics[z].pressure_integral for z in REGION_GROUPS["forefoot"])
        heel_p = sum(zone_metrics[z].pressure_integral for z in REGION_GROUPS["heel"])
        total_ap = fore_p + heel_p
        ap_index = (fore_p - heel_p) / total_ap if total_ap > 1e-8 else 0.0

        # Arch index: midfoot contact relative to total
        mid_contact = sum(zone_metrics[z].contact_area_ratio for z in REGION_GROUPS["midfoot"])
        all_contact = sum(zm.contact_area_ratio for zm in zone_metrics.values())
        arch_index = mid_contact / all_contact if all_contact > 1e-8 else 0.0

        total_pressure = float(p.sum())

        # Pressure symmetry (medial vs lateral)
        symmetry = abs(ml_index)

        return FootAnalysisResult(
            zone_metrics=zone_metrics,
            cop_x=cop_x,
            cop_y=cop_y,
            mediolateral_index=ml_index,
            anteroposterior_index=ap_index,
            arch_index=arch_index,
            total_pressure=total_pressure,
            pressure_symmetry=symmetry,
        )

    def analyze_sequence(self, pressure_seq: np.ndarray) -> dict:
        """Analyze a temporal sequence of pressure frames.

        Args:
            pressure_seq: Shape (T, 1, H, W) or (T, H, W).

        Returns:
            Dict with frame-by-frame results and temporal summaries.
        """
        if pressure_seq.ndim == 4:
            pressure_seq = pressure_seq[:, 0]  # remove channel dim

        frames = [self.analyze_frame(frame) for frame in pressure_seq]

        # Temporal COP trajectory
        cop_trajectory = np.array([(f.cop_x, f.cop_y) for f in frames])

        # COP variability (sway)
        cop_sway = float(np.std(cop_trajectory, axis=0).mean()) if len(frames) > 1 else 0.0

        # Temporal zone summaries
        zone_temporal = {}
        for zone_name in ZONE_DEFINITIONS:
            means = [f.zone_metrics[zone_name].mean_pressure for f in frames]
            peaks = [f.zone_metrics[zone_name].peak_pressure for f in frames]
            zone_temporal[zone_name] = {
                "mean_pressure_avg": float(np.mean(means)),
                "mean_pressure_std": float(np.std(means)),
                "peak_pressure_max": float(np.max(peaks)),
                "peak_pressure_avg": float(np.mean(peaks)),
            }

        ml_indices = [f.mediolateral_index for f in frames]
        ap_indices = [f.anteroposterior_index for f in frames]

        return {
            "frames": frames,
            "cop_trajectory": cop_trajectory,
            "cop_sway": cop_sway,
            "zone_temporal": zone_temporal,
            "ml_index_mean": float(np.mean(ml_indices)),
            "ml_index_std": float(np.std(ml_indices)),
            "ap_index_mean": float(np.mean(ap_indices)),
            "ap_index_std": float(np.std(ap_indices)),
            "num_frames": len(frames),
        }
