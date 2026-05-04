"""Validate WearGait-PD dataset files.

Checks for required IMU and Pressure columns, and reports summary.
"""

import pathlib
import pandas as pd
import numpy as np
from collections import Counter

# Ankle IMU columns
_IMU_COLS = [
    "R_Ankle_Acc_X", "R_Ankle_Acc_Y", "R_Ankle_Acc_Z",
    "R_Ankle_Gyr_X", "R_Ankle_Gyr_Y", "R_Ankle_Gyr_Z",
    "L_Ankle_Acc_X", "L_Ankle_Acc_Y", "L_Ankle_Acc_Z",
    "L_Ankle_Gyr_X", "L_Ankle_Gyr_Y", "L_Ankle_Gyr_Z",
]

# Pressure columns
_PRESSURE_COLS = (
    [f"LPressure{i}" for i in range(1, 17)]
    + [f"RPressure{i}" for i in range(1, 17)]
)

def validate_file(path):
    try:
        # Just read the header first to be fast
        header = pd.read_csv(path, nrows=0).columns.tolist()
        
        has_imu = all(c in header for c in _IMU_COLS)
        has_pressure = all(c in header for c in _PRESSURE_COLS)
        
        # Read a few rows to check for data quality (NaNs)
        df_sample = pd.read_csv(path, nrows=100)
        row_count = len(pd.read_csv(path, usecols=[header[0]])) # Efficient way to count rows
        
        missing_imu = [c for c in _IMU_COLS if c not in header]
        missing_pres = [c for c in _PRESSURE_COLS if c not in header]
        
        return {
            "name": path.name,
            "path": str(path),
            "rows": row_count,
            "has_imu": has_imu,
            "has_pressure": has_pressure,
            "missing_imu_count": len(missing_imu),
            "missing_pres_count": len(missing_pres),
            "error": None
        }
    except Exception as e:
        return {
            "name": path.name,
            "path": str(path),
            "error": str(e)
        }

def main():
    data_dir = pathlib.Path("data/weargait_pd")
    files = sorted(data_dir.glob("**/*.csv"))
    
    # Filter out manifests
    files = [f for f in files if "manifest" not in f.name.lower()]
    
    print(f"Found {len(files)} CSV files.")
    print("-" * 110)
    print(f"{'Filename':<35} | {'Rows':>8} | {'IMU':<5} | {'Pres':<5} | {'Type':<10} | {'Label':<5}")
    print("-" * 110)
    
    results = []
    for f in files:
        res = validate_file(f)
        if res.get("error"):
            print(f"{res['name']:<35} | ERROR: {res['error']}")
            continue
            
        label = "HC" if res['name'].lower().startswith("hc") else "NLS"
        imu_str = "OK" if res['has_imu'] else f"Miss({res['missing_imu_count']})"
        pres_str = "OK" if res['has_pressure'] else f"Miss({res['missing_pres_count']})"
        
        task_type = "SelfPace" if "selfpace" in res['name'].lower() else \
                    "Balance" if "balance" in res['name'].lower() else \
                    "Other"
        
        print(f"{res['name']:<35} | {res['rows']:>8} | {imu_str:<5} | {pres_str:<5} | {task_type:<10} | {label:<5}")
        results.append(res)
    
    print("-" * 110)
    
    # Summary
    if results:
        total = len(results)
        ok_imu = sum(1 for r in results if r['has_imu'])
        ok_pres = sum(1 for r in results if r['has_pressure'])
        hcs = sum(1 for r in results if r['name'].lower().startswith("hc"))
        nls = total - hcs
        
        print(f"Total files: {total}")
        print(f"  Healthy Controls (HC): {hcs}")
        print(f"  Parkinson's (NLS):    {nls}")
        print(f"  IMU OK:               {ok_imu}/{total}")
        print(f"  Pressure OK:          {ok_pres}/{total}")

if __name__ == "__main__":
    main()
