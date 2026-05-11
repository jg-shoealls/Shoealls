"""Compare biomarkers between Healthy Control (HC) and Parkinson's (NLS) data.

This script loads real sample data from WearGait-PD, extracts gait features,
and computes biomarkers to validate the algorithm on real-world data.
"""

import os
import pathlib
import numpy as np
import pandas as pd
import torch
import yaml

from src.analysis.gait_profile import PersonalGaitProfiler
from src.analysis.biomarkers import BiomarkerExtractor
from src.analysis.common import get_feature_korean

# Column definitions
_IMU_COLS = [
    "R_Ankle_Acc_X", "R_Ankle_Acc_Y", "R_Ankle_Acc_Z",
    "R_Ankle_Gyr_X", "R_Ankle_Gyr_Y", "R_Ankle_Gyr_Z",
    "L_Ankle_Acc_X", "L_Ankle_Acc_Y", "L_Ankle_Acc_Z",
    "L_Ankle_Gyr_X", "L_Ankle_Gyr_Y", "L_Ankle_Gyr_Z",
]

_PRESSURE_COLS = (
    [f"LPressure{i}" for i in range(1, 17)]
    + [f"RPressure{i}" for i in range(1, 17)]
)

def load_and_preprocess(path):
    """Load CSV, handle NaNs, and split IMU/Pressure."""
    df = pd.read_csv(path)
    
    # Fill NaNs or drop them. For this demo, we'll fill with 0 
    # but in real use we should drop large NaN blocks.
    df = df.fillna(0)
    
    imu_data = df[_IMU_COLS].values.astype(np.float32)
    pres_data = df[_PRESSURE_COLS].values.astype(np.float32)
    
    # WearGait-PD has 16 sensors per foot.
    # Our analyzer expects a grid. 16 sensors can be 4x4 or 8x2.
    # Let's use 4x8 for combined or 4x4 for single foot.
    # src/analysis/foot_zones.py expects 16x8 by default.
    # We need to adapt the pressure data.
    
    # Reshape 32 columns to 4x8 grid per frame
    # (T, 32) -> (T, 4, 8)
    T = pres_data.shape[0]
    pres_grid = pres_data.reshape(T, 4, 8)
    
    return imu_data, pres_grid

def analyze_sample(name, path, profiler, extractor):
    print(f"\nAnalyzing {name}...")
    imu, pres = load_and_preprocess(path)
    
    # Extract raw features
    # Note: IMU in PersonalGaitProfiler expects (C, T) or (T, C) depending on implementation
    # Based on gait_profile.py: it expects (3, T) or handles (T, 3)
    # We have 12 channels. Let's use R_Ankle (first 3) for simplicity or combine.
    r_accel = imu[:, :3] # (T, 3)
    
    features = profiler.extract_session_features(pres, r_accel)
    
    # Extract biomarkers
    profile = extractor.extract(features)
    
    return features, profile

def main():
    hc_path = "data/weargait_pd/samples/hc100_selfpace.csv"
    nls_path = "data/weargait_pd/samples/nls002_selfpace.csv"
    
    if not os.path.exists(hc_path) or not os.path.exists(nls_path):
        print("Required sample files not found.")
        return

    # Use 4x8 grid for this dataset's 32 pressure sensors
    profiler = PersonalGaitProfiler(grid_h=4, grid_w=8)
    extractor = BiomarkerExtractor()
    
    hc_feats, hc_profile = analyze_sample("Healthy Control (HC100)", hc_path, profiler, extractor)
    nls_feats, nls_profile = analyze_sample("Parkinson's (NLS002)", nls_path, profiler, extractor)
    
    # Comparative Report
    print("\n" + "=" * 80)
    print(f"{'Biomarker':<30} | {'HC Value':>12} | {'NLS Value':>12} | {'Diff %':>8}")
    print("-" * 80)
    
    comparison_metrics = [
        "cadence", "stride_regularity", "step_symmetry", 
        "acceleration_rms", "cop_sway", "arch_index"
    ]
    
    for m in comparison_metrics:
        v_hc = hc_feats.get(m, 0)
        v_nls = nls_feats.get(m, 0)
        k_name = get_feature_korean(m)
        
        diff_pct = (v_nls - v_hc) / (v_hc + 1e-8) * 100
        print(f"{k_name:<30} | {v_hc:>12.4f} | {v_nls:>12.4f} | {diff_pct:>7.1f}%")
    
    print("-" * 80)
    print(f"HC Abnormal Markers: {hc_profile.abnormal_count}/{hc_profile.total_count}")
    print(f"NLS Abnormal Markers: {nls_profile.abnormal_count}/{nls_profile.total_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()
