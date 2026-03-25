"""데이터 검증 스크립트.

수집한 데이터가 알고리즘 입력 형식에 맞는지 확인합니다.
학습 전에 반드시 이 스크립트를 먼저 실행하세요.

실행:
    python scripts/validate_data.py --data-dir data/collected/

출력 예시:
    ✓ 20 subjects found
    ✓ IMU: (T, 6) shape OK — mean T=245, range [180, 310]
    ✓ Pressure: (T, 128) shape OK — grid 16x8
    ✓ Skeleton: (T, 51) shape OK — 17 joints × 3 coords
    ✓ Labels: 4 classes, balanced [5, 5, 5, 5]
    ✗ subject_015: IMU has 5 columns, expected 6
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def check_csv(filepath, expected_cols=None, name=""):
    """CSV 파일 검증."""
    issues = []
    try:
        # Try to detect header
        with open(filepath) as f:
            first_line = f.readline().strip()

        try:
            float(first_line.split(",")[0])
            has_header = False
        except ValueError:
            has_header = True

        data = np.loadtxt(filepath, delimiter=",", skiprows=1 if has_header else 0)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if expected_cols and data.shape[1] != expected_cols:
            issues.append(
                f"{name}: {data.shape[1]} columns, expected {expected_cols}"
            )

        if data.shape[0] < 30:
            issues.append(f"{name}: only {data.shape[0]} rows (minimum 30 recommended)")

        if np.isnan(data).any():
            nan_count = np.isnan(data).sum()
            issues.append(f"{name}: {nan_count} NaN values found")

        if np.isinf(data).any():
            issues.append(f"{name}: Inf values found")

        return data.shape, issues

    except Exception as e:
        return None, [f"{name}: Failed to load — {e}"]


def main():
    parser = argparse.ArgumentParser(description="데이터 형식 검증")
    parser.add_argument("--data-dir", type=str, required=True, help="데이터 폴더 경로")
    parser.add_argument("--pressure-grid", type=str, default="16,8",
                        help="족저압 센서 그리드 크기 (H,W)")
    parser.add_argument("--num-joints", type=int, default=17, help="스켈레톤 관절 수")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    h, w = map(int, args.pressure_grid.split(","))
    j = args.num_joints

    print("=" * 60)
    print("멀티모달 보행 데이터 검증")
    print("=" * 60)
    print(f"데이터 경로: {data_dir}")
    print(f"족저압 그리드: {h}x{w} = {h*w}")
    print(f"스켈레톤 관절: {j} ({j*3} columns)")
    print()

    # Find subjects
    subject_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not subject_dirs:
        print("ERROR: No subject directories found!")
        print(f"Expected: {data_dir}/subject_001/imu.csv, etc.")
        return

    print(f"{'Subjects found':.<40} {len(subject_dirs)}")

    # Check each subject
    all_issues = []
    imu_shapes, pressure_shapes, skeleton_shapes = [], [], []

    for subject_dir in subject_dirs:
        # IMU
        imu_path = subject_dir / "imu.csv"
        if imu_path.exists():
            shape, issues = check_csv(imu_path, expected_cols=6, name=f"{subject_dir.name}/imu")
            if shape:
                imu_shapes.append(shape)
            all_issues.extend(issues)
        else:
            all_issues.append(f"{subject_dir.name}: imu.csv not found")

        # Pressure
        pressure_path = subject_dir / "pressure.csv"
        if pressure_path.exists():
            shape, issues = check_csv(pressure_path, expected_cols=h*w, name=f"{subject_dir.name}/pressure")
            if shape:
                pressure_shapes.append(shape)
            all_issues.extend(issues)
        else:
            all_issues.append(f"{subject_dir.name}: pressure.csv not found")

        # Skeleton
        skeleton_path = subject_dir / "skeleton.csv"
        if skeleton_path.exists():
            expected = j * 3
            shape, issues = check_csv(skeleton_path, expected_cols=expected, name=f"{subject_dir.name}/skeleton")
            if shape:
                skeleton_shapes.append(shape)
            all_issues.extend(issues)
        else:
            # Try 2D skeleton
            shape, issues = check_csv(
                subject_dir / "skeleton.csv", expected_cols=j*2,
                name=f"{subject_dir.name}/skeleton(2D)"
            )
            if shape:
                skeleton_shapes.append(shape)
                print(f"  NOTE: {subject_dir.name} has 2D skeleton (will pad z=0)")
            else:
                all_issues.append(f"{subject_dir.name}: skeleton.csv not found")

    # Summary
    print()
    if imu_shapes:
        lengths = [s[0] for s in imu_shapes]
        print(f"{'IMU':.<40} OK ({len(imu_shapes)} files)")
        print(f"  Shape per file: (T, 6)")
        print(f"  Frame counts: mean={np.mean(lengths):.0f}, "
              f"min={min(lengths)}, max={max(lengths)}")

    if pressure_shapes:
        lengths = [s[0] for s in pressure_shapes]
        print(f"{'Pressure':.<40} OK ({len(pressure_shapes)} files)")
        print(f"  Shape per file: (T, {h*w}) -> reshape to (T, {h}, {w})")
        print(f"  Frame counts: mean={np.mean(lengths):.0f}, "
              f"min={min(lengths)}, max={max(lengths)}")

    if skeleton_shapes:
        lengths = [s[0] for s in skeleton_shapes]
        cols = skeleton_shapes[0][1]
        dim = "3D" if cols == j * 3 else "2D"
        print(f"{'Skeleton':.<40} OK ({len(skeleton_shapes)} files, {dim})")
        print(f"  Shape per file: (T, {cols})")
        print(f"  Frame counts: mean={np.mean(lengths):.0f}, "
              f"min={min(lengths)}, max={max(lengths)}")

    # Labels
    label_path = data_dir / "labels.csv"
    if label_path.exists():
        import pandas as pd
        df = pd.read_csv(label_path)
        label_col = df.columns[1]
        counts = Counter(df[label_col])
        print(f"\n{'Labels':.<40} OK")
        print(f"  File: {label_path}")
        print(f"  Classes: {len(counts)}")
        for label, count in sorted(counts.items()):
            print(f"    class {label}: {count} samples")
    else:
        all_issues.append("labels.csv not found in data root")

    # Issues
    print()
    if all_issues:
        print(f"ISSUES FOUND: {len(all_issues)}")
        for issue in all_issues:
            print(f"  [!] {issue}")
    else:
        print("ALL CHECKS PASSED!")

    # Recommendation
    print()
    print("-" * 60)
    print("다음 단계:")
    if not all_issues:
        print(f"  python run_real_data.py --data-dir {data_dir} --format folder")
    else:
        print("  위 이슈를 수정한 뒤 다시 검증해 주세요.")


if __name__ == "__main__":
    main()
