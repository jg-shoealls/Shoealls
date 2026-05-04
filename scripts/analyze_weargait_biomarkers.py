"""Analyze WearGait-PD sample distributions and gait biomarkers.

Outputs:
  outputs/weargait_biomarker_analysis/distribution_summary.csv
  outputs/weargait_biomarker_analysis/biomarker_summary.csv
  outputs/weargait_biomarker_analysis/cohort_task_summary.csv
  outputs/weargait_biomarker_analysis/cohort_task_summary.md
  outputs/weargait_biomarker_analysis/hc100_vs_nls002_selfpace.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/weargait_pd")
OUT_DIR = Path("outputs/weargait_biomarker_analysis")

BASE_NORMAL_RANGES = {
    "gait_speed": (1.0, 1.4),
    "cadence": (100.0, 130.0),
    "stride_regularity": (0.7, 1.0),
    "step_symmetry": (0.85, 1.0),
    "cop_sway": (0.0, 0.06),
    "ml_variability": (0.0, 0.10),
    "heel_pressure_ratio": (0.25, 0.40),
    "forefoot_pressure_ratio": (0.35, 0.55),
    "arch_index": (0.15, 0.35),
    "pressure_asymmetry": (0.0, 0.12),
    "acceleration_rms": (0.8, 2.5),
    "acceleration_variability": (0.0, 0.35),
    "trunk_sway": (0.0, 3.0),
}

IMU_SUFFIXES = (
    "_Acc_X", "_Acc_Y", "_Acc_Z",
    "_FreeAcc_E", "_FreeAcc_N", "_FreeAcc_U",
    "_Gyr_X", "_Gyr_Y", "_Gyr_Z",
)


def parse_seconds(series: pd.Series) -> np.ndarray:
    text = series.astype(str).str.replace(" sec", "", regex=False)
    values = pd.to_numeric(text, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(values).sum() < 2:
        return np.arange(len(series), dtype=float) / 100.0
    return values


def sample_rate_from_time(t: np.ndarray) -> float:
    diffs = np.diff(t[np.isfinite(t)])
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.size == 0:
        return 100.0
    return float(1.0 / np.median(diffs))


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce")


def has_gait_channels(path: Path) -> bool:
    try:
        cols = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    return "Time" in cols and any(c.endswith("_Acc_X") for c in cols)


def classify_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    imu_cols = [c for c in df.columns if c.endswith(IMU_SUFFIXES)]
    pressure_cols = (
        [c for c in df.columns if c.startswith("LPressure") or c.startswith("RPressure")]
        + [c for c in ["LTotalForce", "RTotalForce", "L Foot Pressure", "R Foot Pressure"] if c in df.columns]
    )
    return imu_cols, pressure_cols


def summarize_distribution(path: Path, df: pd.DataFrame) -> list[dict[str, object]]:
    imu_cols, pressure_cols = classify_columns(df)
    rows = []
    for sensor_type, cols in [("imu", imu_cols), ("pressure", pressure_cols)]:
        if not cols:
            continue
        vals = numeric(df, cols).to_numpy(dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append({
            "file": str(path),
            "sensor_type": sensor_type,
            "channels": len(cols),
            "count": int(vals.size),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        })
    return rows


def vector_magnitude(df: pd.DataFrame, cols: list[str]) -> np.ndarray | None:
    if not all(c in df.columns for c in cols):
        return None
    arr = numeric(df, cols).to_numpy(dtype=float)
    if arr.size == 0:
        return None
    return np.sqrt(np.nansum(arr ** 2, axis=1))


def autocorr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return np.array([])
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom <= 1e-12:
        return np.array([])
    ac = np.correlate(x, x, mode="full")[x.size - 1:]
    return ac / denom


def regularity_metrics(acc_mag: np.ndarray, sr: float) -> tuple[float, float]:
    ac = autocorr(acc_mag)
    if ac.size == 0:
        return np.nan, np.nan
    min_lag = max(2, int(0.30 * sr))
    max_lag = min(ac.size - 1, int(2.00 * sr))
    peaks: list[tuple[int, float]] = []
    for i in range(min_lag, max_lag):
        if ac[i] > ac[i - 1] and ac[i] > ac[i + 1]:
            peaks.append((i, float(ac[i])))
    if not peaks:
        return np.nan, np.nan
    peaks = sorted(peaks, key=lambda p: p[1], reverse=True)
    stride_regularity = max(0.0, min(1.0, peaks[0][1]))
    if len(peaks) < 2:
        return stride_regularity, np.nan
    a, b = peaks[0][1], peaks[1][1]
    step_symmetry = min(a, b) / (max(a, b) + 1e-12)
    return stride_regularity, float(max(0.0, min(1.0, step_symmetry)))


def estimate_cadence(df: pd.DataFrame, acc_mag: np.ndarray, sr: float, duration_s: float) -> float:
    contact_cols = [c for c in ["L Foot Contact", "R Foot Contact"] if c in df.columns]
    if contact_cols:
        contact = numeric(df, contact_cols).fillna(0).to_numpy(dtype=float)
        rising = np.diff((contact > 0).astype(int), axis=0) == 1
        steps = int(np.sum(rising))
        if steps >= 4 and duration_s > 0:
            return float(steps / duration_s * 60.0)

    x = pd.Series(acc_mag).interpolate(limit_direction="both").to_numpy(dtype=float)
    x = x - np.nanmedian(x)
    threshold = np.nanmean(x) + 0.75 * np.nanstd(x)
    min_dist = max(1, int(0.30 * sr))
    peaks = []
    last = -min_dist
    for i in range(1, len(x) - 1):
        if i - last < min_dist:
            continue
        if x[i] > threshold and x[i] > x[i - 1] and x[i] >= x[i + 1]:
            peaks.append(i)
            last = i
    if duration_s <= 0:
        return np.nan
    return float(len(peaks) / duration_s * 60.0)


def pressure_features(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if {"LTotalForce", "RTotalForce"}.issubset(df.columns):
        left = pd.to_numeric(df["LTotalForce"], errors="coerce")
        right = pd.to_numeric(df["RTotalForce"], errors="coerce")
    elif {"L Foot Pressure", "R Foot Pressure"}.issubset(df.columns):
        left = pd.to_numeric(df["L Foot Pressure"], errors="coerce")
        right = pd.to_numeric(df["R Foot Pressure"], errors="coerce")
    else:
        left = right = None

    if left is not None and right is not None:
        left_sum = float(left.fillna(0).sum())
        right_sum = float(right.fillna(0).sum())
        denom = left_sum + right_sum
        out["pressure_asymmetry"] = abs(left_sum - right_sum) / denom if denom > 0 else np.nan

    l_cols = [f"LPressure{i}" for i in range(1, 17) if f"LPressure{i}" in df.columns]
    r_cols = [f"RPressure{i}" for i in range(1, 17) if f"RPressure{i}" in df.columns]
    if len(l_cols) == 16 and len(r_cols) == 16:
        arr = numeric(df, l_cols + r_cols).to_numpy(dtype=float).reshape(len(df), 2, 4, 4)
        total = np.nansum(arr, axis=(1, 2, 3))
        heel = np.nansum(arr[:, :, 3, :], axis=(1, 2))
        fore = np.nansum(arr[:, :, 0:2, :], axis=(1, 2, 3))
        mid = np.nansum(arr[:, :, 2, :], axis=(1, 2))
        valid = total > 0
        if np.any(valid):
            out["heel_pressure_ratio"] = float(np.nanmean(heel[valid] / total[valid]))
            out["forefoot_pressure_ratio"] = float(np.nanmean(fore[valid] / total[valid]))
            out["arch_index"] = float(np.nanmean(mid[valid] / total[valid]))

        cop_cols = [c for c in ["LCoP_X", "LCoP_Y", "RCoP_X", "RCoP_Y"] if c in df.columns]
        if cop_cols:
            cop = numeric(df, cop_cols).replace(0, np.nan)
            out["cop_sway"] = float(cop.std(skipna=True).mean())
    return out


def range_status(metric: str, value: float) -> str:
    if not np.isfinite(value) or metric not in BASE_NORMAL_RANGES:
        return "NA"
    low, high = BASE_NORMAL_RANGES[metric]
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "normal"


def infer_group(subject: str) -> str:
    subject = subject.lower()
    if subject.startswith(("hc", "whc")):
        return "control"
    if subject.startswith(("nls", "wpd")):
        return "parkinson"
    return "unknown"


def extract_biomarkers(path: Path, df: pd.DataFrame) -> dict[str, object]:
    t = parse_seconds(df["Time"]) if "Time" in df.columns else np.arange(len(df)) / 100.0
    sr = sample_rate_from_time(t)
    duration_s = float(np.nanmax(t) - np.nanmin(t)) if len(t) else np.nan

    acc_mag = vector_magnitude(df, ["LowerBack_FreeAcc_E", "LowerBack_FreeAcc_N", "LowerBack_FreeAcc_U"])
    if acc_mag is None:
        acc_mag = vector_magnitude(df, ["LowerBack_Acc_X", "LowerBack_Acc_Y", "LowerBack_Acc_Z"])
    if acc_mag is None:
        acc_mag = np.array([], dtype=float)

    p = pressure_features(df)
    stride_regularity, step_symmetry = regularity_metrics(acc_mag, sr)
    acceleration_rms = float(np.sqrt(np.nanmean(acc_mag ** 2))) if acc_mag.size else np.nan
    acceleration_variability = float(np.nanstd(acc_mag) / (np.nanmean(acc_mag) + 1e-12)) if acc_mag.size else np.nan
    cadence = estimate_cadence(df, acc_mag, sr, duration_s) if acc_mag.size else np.nan

    values = {
        "cadence": cadence,
        "stride_regularity": stride_regularity,
        "step_symmetry": step_symmetry,
        "acceleration_rms": acceleration_rms,
        "acceleration_variability": acceleration_variability,
        "trunk_sway": acceleration_rms * 1.2 if np.isfinite(acceleration_rms) else np.nan,
        "gait_speed": cadence / 60.0 * 0.375 if np.isfinite(cadence) else np.nan,
        "ml_variability": p.get("cop_sway", np.nan) * 1.5 if np.isfinite(p.get("cop_sway", np.nan)) else np.nan,
        **p,
    }

    row = {
        "file": str(path),
        "subject": path.stem.split("_")[0],
        "task": "_".join(path.stem.split("_")[1:]),
        "duration_s": duration_s,
        "sample_rate_hz": sr,
    }
    row["group"] = infer_group(str(row["subject"]))
    for metric in BASE_NORMAL_RANGES:
        value = values.get(metric, np.nan)
        row[metric] = value
        row[f"{metric}_status"] = range_status(metric, value)
    return row


def fmt(value: object) -> str:
    if value is None:
        return "NA"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.4f}"


def write_pair_report(biomarkers: pd.DataFrame, out_dir: Path) -> None:
    pair = biomarkers[
        biomarkers["file"].str.endswith("samples\\hc100_selfpace.csv")
        | biomarkers["file"].str.endswith("samples/nls002_selfpace.csv")
        | biomarkers["file"].str.endswith("samples\\nls002_selfpace.csv")
    ].copy()
    hc = pair[pair["subject"].eq("hc100")].head(1)
    nls = pair[pair["subject"].eq("nls002")].head(1)
    lines = [
        "# HC100 vs NLS002 SelfPace Biomarker Check",
        "",
        "Normal ranges are the base biomarker ranges used by `configs/biomarker_analysis.yaml`/the analysis module.",
        "",
        "| metric | normal range | hc100 | hc status | nls002 | nls status | nls-hc % |",
        "|---|---:|---:|---|---:|---|---:|",
    ]
    for metric in BASE_NORMAL_RANGES:
        low, high = BASE_NORMAL_RANGES[metric]
        hc_v = hc.iloc[0][metric] if len(hc) else np.nan
        nls_v = nls.iloc[0][metric] if len(nls) else np.nan
        if np.isfinite(hc_v) and np.isfinite(nls_v) and abs(hc_v) > 1e-12:
            diff = (nls_v - hc_v) / abs(hc_v) * 100.0
        else:
            diff = np.nan
        lines.append(
            f"| {metric} | [{low:g}, {high:g}] | {fmt(hc_v)} | "
            f"{hc.iloc[0][metric + '_status'] if len(hc) else 'NA'} | "
            f"{fmt(nls_v)} | {nls.iloc[0][metric + '_status'] if len(nls) else 'NA'} | {fmt(diff)} |"
        )
    lines.extend([
        "",
        "Notes:",
        "- HC100 SelfPace has coarse left/right foot-pressure columns but no insole pressure grid, so zone pressure ratios are `NA` for HC100.",
        "- `gait_speed` is a cadence-derived proxy because walkway distance events are not available in these sample CSVs.",
    ])
    (out_dir / "hc100_vs_nls002_selfpace.md").write_text("\n".join(lines), encoding="utf-8")


def write_cohort_summary(biomarkers: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    grouped = biomarkers.groupby(["group", "task"], dropna=False)
    for (group, task), frame in grouped:
        row: dict[str, object] = {
            "group": group,
            "task": task,
            "files": int(len(frame)),
            "subjects": int(frame["subject"].nunique()),
        }
        for metric in BASE_NORMAL_RANGES:
            values = pd.to_numeric(frame[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=0)) if values.notna().any() else np.nan
            status = frame[f"{metric}_status"]
            known = status.ne("NA")
            row[f"{metric}_abnormal_rate"] = (
                float(status[known].ne("normal").mean()) if known.any() else np.nan
            )
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["group", "task"])
    summary.to_csv(out_dir / "cohort_task_summary.csv", index=False)

    key_metrics = [
        "cadence",
        "stride_regularity",
        "step_symmetry",
        "cop_sway",
        "pressure_asymmetry",
        "acceleration_variability",
    ]
    lines = [
        "# WearGait-PD Cohort/Task Biomarker Summary",
        "",
        "Values are mean +- std by inferred group and task. Abnormal rate is the fraction outside the configured normal range.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.extend([
            f"## {row['group']} / {row['task']}",
            "",
            f"- files: {int(row['files'])}",
            f"- subjects: {int(row['subjects'])}",
            "",
            "| metric | mean +- std | abnormal rate |",
            "|---|---:|---:|",
        ])
        for metric in key_metrics:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            abnormal = row[f"{metric}_abnormal_rate"]
            lines.append(
                f"| {metric} | {fmt(mean)} +- {fmt(std)} | {fmt(abnormal)} |"
            )
        lines.append("")
    (out_dir / "cohort_task_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(data_dir.rglob("*.csv"))
    gait_paths = [p for p in csv_paths if has_gait_channels(p)]

    distribution_rows = []
    biomarker_rows = []
    skipped = []
    for path in csv_paths:
        if path not in gait_paths:
            skipped.append(str(path))
            continue
        df = pd.read_csv(path, low_memory=False)
        distribution_rows.extend(summarize_distribution(path, df))
        biomarker_rows.append(extract_biomarkers(path, df))

    distribution = pd.DataFrame(distribution_rows)
    biomarkers = pd.DataFrame(biomarker_rows)
    distribution.to_csv(out_dir / "distribution_summary.csv", index=False)
    biomarkers.to_csv(out_dir / "biomarker_summary.csv", index=False)
    pd.Series(skipped, name="skipped_non_gait_csv").to_csv(out_dir / "skipped_files.csv", index=False)
    write_cohort_summary(biomarkers, out_dir)
    write_pair_report(biomarkers, out_dir)

    print(f"CSV files found: {len(csv_paths)}")
    print(f"Gait CSV files analyzed: {len(gait_paths)}")
    print(f"Non-gait CSV files skipped: {len(skipped)}")
    print(f"Wrote: {out_dir / 'distribution_summary.csv'}")
    print(f"Wrote: {out_dir / 'biomarker_summary.csv'}")
    print(f"Wrote: {out_dir / 'cohort_task_summary.csv'}")
    print(f"Wrote: {out_dir / 'cohort_task_summary.md'}")
    print(f"Wrote: {out_dir / 'hc100_vs_nls002_selfpace.md'}")


if __name__ == "__main__":
    main()
