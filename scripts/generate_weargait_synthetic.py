"""Generate synthetic WearGait-PD format CSV files for pipeline validation.

Produces HC (healthy control) and PD (Parkinson's Disease) gait recordings
that match the column schema expected by WearGaitDataset:
  IMU (12 cols): R_Ankle_{Acc,Gyr}_{X,Y,Z}  +  L_Ankle_{Acc,Gyr}_{X,Y,Z}
  Pressure (32 cols): LPressure1-16 + RPressure1-16

File naming:
  HC  → hcXXX_selfpace.csv   (label=0)
  PD  → nlsXXX_selfpace.csv  (label=1)

Usage:
  python scripts/generate_weargait_synthetic.py --mode smoke   # 4 subjects
  python scripts/generate_weargait_synthetic.py --mode full    # 20 subjects
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FS = 100.0  # Hz — matches WearGait-PD sample rate

IMU_COLS = [
    "R_Ankle_Acc_X", "R_Ankle_Acc_Y", "R_Ankle_Acc_Z",
    "R_Ankle_Gyr_X", "R_Ankle_Gyr_Y", "R_Ankle_Gyr_Z",
    "L_Ankle_Acc_X", "L_Ankle_Acc_Y", "L_Ankle_Acc_Z",
    "L_Ankle_Gyr_X", "L_Ankle_Gyr_Y", "L_Ankle_Gyr_Z",
]
PRESSURE_COLS = (
    [f"LPressure{i}" for i in range(1, 17)]
    + [f"RPressure{i}" for i in range(1, 17)]
)


# ─────────────────────────────────────────────────────────────────────
# Gait signal generators
# ─────────────────────────────────────────────────────────────────────

def _hc_imu(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Healthy-control ankle IMU (12, n_samples).

    - Regular 2 Hz stride cycle, low stride-to-stride variability
    - No resting tremor in 4-6 Hz band
    - Left/right mildly different but symmetric overall
    """
    t = np.arange(n_samples) / FS

    # Vertical acceleration: clean 2 Hz sinusoid (± small Gaussian noise)
    base_stride = np.sin(2 * np.pi * 2.0 * t)

    imu = np.zeros((12, n_samples), dtype=np.float32)

    for side in range(2):          # 0=right, 1=left
        offset = side * 6
        phase_offset = np.pi * side  # left/right half-cycle apart

        # Z (vertical, main propulsion)
        imu[offset + 2] = (
            1.0 * np.sin(2 * np.pi * 2.0 * t + phase_offset)
            + rng.normal(0, 0.05, n_samples)
        ).astype(np.float32)

        # X (antero-posterior)
        imu[offset + 0] = (
            0.3 * np.sin(2 * np.pi * 2.0 * t + phase_offset + 0.5)
            + rng.normal(0, 0.03, n_samples)
        ).astype(np.float32)

        # Y (medio-lateral)
        imu[offset + 1] = (
            0.15 * np.sin(2 * np.pi * 2.0 * t + phase_offset + 1.0)
            + rng.normal(0, 0.02, n_samples)
        ).astype(np.float32)

        # Gyroscope (angular velocity, ~10 deg/s peak)
        imu[offset + 3] = (0.15 * np.cos(2 * np.pi * 2.0 * t + phase_offset) + rng.normal(0, 0.02, n_samples)).astype(np.float32)
        imu[offset + 4] = (0.08 * np.cos(2 * np.pi * 2.0 * t + phase_offset) + rng.normal(0, 0.015, n_samples)).astype(np.float32)
        imu[offset + 5] = (0.05 * np.cos(2 * np.pi * 2.0 * t + phase_offset) + rng.normal(0, 0.01, n_samples)).astype(np.float32)

    return imu


def _pd_imu(n_samples: int, rng: np.random.Generator, severity: float = 0.5) -> np.ndarray:
    """Parkinson's Disease ankle IMU (12, n_samples).

    Severity ∈ [0, 1] controls how pronounced the PD features are.
    - Irregular stride: cumulative phase drift causes stride CV > 0.06
    - Bradykinesia: reduced vertical acceleration amplitude (0.4-0.7 m/s²)
    - Resting tremor: 4-6 Hz component on X/Y axes (amplitude 0.3-0.5)
    - Left/right asymmetry stronger than HC
    """
    t = np.arange(n_samples) / FS

    # Cumulative phase drift (stride irregularity)
    phase_drift = np.cumsum(rng.normal(0, 0.15 + 0.2 * severity, n_samples)).astype(np.float64)

    # Bradykinesia reduces amplitude
    amplitude = 0.70 - 0.30 * severity  # 0.40 – 0.70

    # Tremor amplitude
    tremor_amp = 0.30 + 0.20 * severity  # 0.30 – 0.50
    tremor_freq = rng.uniform(4.5, 5.5)  # 4.5–5.5 Hz resting tremor

    imu = np.zeros((12, n_samples), dtype=np.float32)

    for side in range(2):
        offset = side * 6
        phase_offset = np.pi * side

        # Z – irregular stride cycle
        imu[offset + 2] = (
            amplitude * np.sin(2 * np.pi * 2.0 * t + phase_offset + phase_drift)
            + rng.normal(0, 0.08, n_samples)
        ).astype(np.float32)

        # X – resting tremor (4-6 Hz) + gait component
        imu[offset + 0] = (
            0.2 * np.sin(2 * np.pi * 2.0 * t + phase_offset + 0.5 + phase_drift)
            + tremor_amp * np.sin(2 * np.pi * tremor_freq * t + rng.uniform(0, 2 * np.pi))
            + rng.normal(0, 0.05, n_samples)
        ).astype(np.float32)

        # Y – resting tremor
        imu[offset + 1] = (
            0.10 * np.sin(2 * np.pi * 2.0 * t + phase_offset + 1.0 + phase_drift)
            + tremor_amp * np.cos(2 * np.pi * tremor_freq * t + rng.uniform(0, 2 * np.pi))
            + rng.normal(0, 0.04, n_samples)
        ).astype(np.float32)

        # Gyroscope – higher variability
        imu[offset + 3] = (
            0.20 * np.cos(2 * np.pi * 2.0 * t + phase_offset + phase_drift)
            + rng.normal(0, 0.04, n_samples)
        ).astype(np.float32)
        imu[offset + 4] = (
            0.10 * np.cos(2 * np.pi * 2.0 * t + phase_offset + phase_drift)
            + rng.normal(0, 0.03, n_samples)
        ).astype(np.float32)
        imu[offset + 5] = (
            0.06 * np.cos(2 * np.pi * 2.0 * t + phase_offset + phase_drift)
            + rng.normal(0, 0.02, n_samples)
        ).astype(np.float32)

    return imu


def _hc_pressure(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Healthy-control plantar pressure (n_samples, 32).

    Regular alternating loading: left foot (col 0-15) and right foot (col 16-31).
    Symmetric total load. Pressure mapped to anatomical 4×4 grid:
      rows 0-1: forefoot, row 2: midfoot, row 3: heel
    """
    t = np.arange(n_samples) / FS
    pressure = np.zeros((n_samples, 32), dtype=np.float32)

    stride_period = 0.5   # s (2 Hz stride → 0.5s per step)
    for i in range(n_samples):
        phase = (t[i] % stride_period) / stride_period  # 0–1

        # Left foot active first 50%, right foot second 50%
        # Smooth half-sine loading curve
        if phase < 0.48:
            left_load  = np.clip(np.sin(np.pi * phase / 0.48), 0, 1)
            right_load = 0.0
        else:
            left_load  = 0.0
            right_load = np.clip(np.sin(np.pi * (phase - 0.48) / 0.52), 0, 1)

        base_left  = rng.uniform(0.4, 0.8, 16).astype(np.float32) * left_load
        base_right = rng.uniform(0.4, 0.8, 16).astype(np.float32) * right_load

        # Heel emphasis (rows 3 in 4×4 = index 12-15)
        base_left[12:16]  *= 1.4
        base_right[12:16] *= 1.4

        pressure[i, :16]  = base_left
        pressure[i, 16:]  = base_right

    return pressure


def _pd_pressure(n_samples: int, rng: np.random.Generator, severity: float = 0.5) -> np.ndarray:
    """PD plantar pressure (n_samples, 32).

    - Left/right asymmetry (one side carries more load)
    - Increased double-support frames (both feet loaded simultaneously)
    - Reduced heel-strike on affected side (forefoot-heavy)
    """
    t = np.arange(n_samples) / FS
    pressure = np.zeros((n_samples, 32), dtype=np.float32)

    # Asymmetry: affected side is 'right' here (random per subject)
    asymmetry = 0.15 + 0.15 * severity   # right carries 15-30% less
    double_support_extra = 0.08 + 0.12 * severity  # extra overlap fraction

    stride_period = 0.5

    for i in range(n_samples):
        phase = (t[i] % stride_period) / stride_period

        # Longer double-support: extend overlap window
        ds = double_support_extra
        if phase < (0.50 - ds):
            left_load  = np.clip(np.sin(np.pi * phase / (0.50 - ds)), 0, 1)
            right_load = 0.0
        elif phase < (0.50 + ds):
            # Both feet loaded → double support
            left_load  = 0.5
            right_load = 0.5 * (1 - asymmetry)
        else:
            left_load  = 0.0
            right_load = np.clip(np.sin(np.pi * (phase - (0.50 + ds)) / (0.50 - ds)), 0, 1) * (1 - asymmetry)

        base_left  = rng.uniform(0.4, 0.8, 16).astype(np.float32) * left_load
        base_right = rng.uniform(0.4, 0.8, 16).astype(np.float32) * right_load

        # Reduced heel on affected (right) side → forefoot heavy
        base_left[12:16]  *= 1.3
        base_right[:4]    *= 1.5   # forefoot heavy on right
        base_right[12:16] *= 0.6   # reduced heel

        pressure[i, :16]  = base_left
        pressure[i, 16:]  = base_right

    return pressure


# ─────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────

def _write_csv(path: Path, imu: np.ndarray, pressure: np.ndarray) -> None:
    """Write a single WearGait-format CSV.

    imu:      (12, n_samples)
    pressure: (n_samples, 32)
    """
    imu_df       = pd.DataFrame(imu.T, columns=IMU_COLS)
    pressure_df  = pd.DataFrame(pressure, columns=PRESSURE_COLS)
    df = pd.concat([imu_df, pressure_df], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")


# ─────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────

def generate(
    out_dir: str | Path,
    n_hc: int,
    n_pd: int,
    n_samples: int,
    seed: int = 42,
) -> list[Path]:
    """Generate n_hc HC and n_pd PD WearGait CSV files.

    Returns list of generated file paths.
    """
    out_dir = Path(out_dir)
    rng_root = np.random.default_rng(seed)
    paths: list[Path] = []

    for i in range(n_hc):
        rng = np.random.default_rng(rng_root.integers(0, 2**32))
        imu      = _hc_imu(n_samples, rng)
        pressure = _hc_pressure(n_samples, rng)
        subject_id = f"hc{i + 1:03d}"
        p = out_dir / f"{subject_id}_selfpace.csv"
        _write_csv(p, imu, pressure)
        paths.append(p)
        print(f"  [HC]  {p.name}  ({n_samples} rows)")

    for i in range(n_pd):
        rng = np.random.default_rng(rng_root.integers(0, 2**32))
        severity = float(rng.uniform(0.3, 0.9))
        imu      = _pd_imu(n_samples, rng, severity=severity)
        pressure = _pd_pressure(n_samples, rng, severity=severity)
        subject_id = f"nls{i + 1:03d}"
        p = out_dir / f"{subject_id}_selfpace.csv"
        _write_csv(p, imu, pressure)
        paths.append(p)
        print(f"  [PD]  {p.name}  ({n_samples} rows, severity={severity:.2f})")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic WearGait-PD CSV files.")
    parser.add_argument(
        "--mode", choices=["smoke", "full"], default="full",
        help="smoke: 4 subjects (quick test), full: 20 subjects (LOSO benchmark)",
    )
    parser.add_argument("--out-dir", default="data/weargait_pd", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.mode == "smoke":
        n_hc, n_pd, n_samples = 2, 2, 4_000
    else:
        n_hc, n_pd, n_samples = 10, 10, 6_000

    print(f"Mode: {args.mode} | HC={n_hc}, PD={n_pd}, rows/file={n_samples}")
    print(f"Output: {args.out_dir}")
    paths = generate(args.out_dir, n_hc, n_pd, n_samples, seed=args.seed)
    print(f"\nGenerated {len(paths)} files in {args.out_dir}/")


if __name__ == "__main__":
    main()
