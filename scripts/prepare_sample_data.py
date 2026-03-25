"""샘플 데이터 생성 스크립트.

실제 데이터 수집 전에 파이프라인을 검증하기 위한
폴더 구조 샘플 데이터를 생성합니다.

실행:
    python scripts/prepare_sample_data.py

결과:
    data/sample/
    ├── subject_001/
    │   ├── imu.csv
    │   ├── pressure.csv
    │   └── skeleton.csv
    ├── subject_002/
    │   └── ...
    └── labels.csv
"""

from pathlib import Path

import numpy as np

from src.data.synthetic import (
    generate_synthetic_imu,
    generate_synthetic_pressure,
    generate_synthetic_skeleton,
)


def main():
    output_dir = Path("data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    num_subjects_per_class = 5
    num_classes = 4
    class_names = ["normal", "antalgic", "ataxic", "parkinsonian"]

    labels_rows = []
    subject_id = 1

    for class_idx in range(num_classes):
        for _ in range(num_subjects_per_class):
            sid = f"subject_{subject_id:03d}"
            subject_dir = output_dir / sid
            subject_dir.mkdir(exist_ok=True)

            num_frames = 200 + rng.integers(-20, 20)

            # IMU: (T, 6) with header
            imu = generate_synthetic_imu(num_frames, class_idx, rng)
            header = "ax,ay,az,gx,gy,gz"
            np.savetxt(subject_dir / "imu.csv", imu, delimiter=",",
                       header=header, comments="")

            # Pressure: (T, 128) = flattened 16x8 grid
            pressure = generate_synthetic_pressure(
                num_frames, class_idx, (16, 8), rng
            )
            flat_pressure = pressure.reshape(num_frames, -1)
            cols = [f"p{i}" for i in range(128)]
            np.savetxt(subject_dir / "pressure.csv", flat_pressure, delimiter=",",
                       header=",".join(cols), comments="")

            # Skeleton: (T, 51) = 17 joints × 3 coords
            skeleton = generate_synthetic_skeleton(num_frames, class_idx, 17, rng)
            flat_skeleton = skeleton.reshape(num_frames, -1)
            joint_names = ["hip", "spine", "head", "Lshoulder", "Lelbow", "Lwrist",
                           "Rshoulder", "Relbow", "Rwrist", "Lhip", "Lknee", "Lankle",
                           "Rhip", "Rknee", "Rankle", "Lfoot", "Rfoot"]
            cols = [f"{j}_{c}" for j in joint_names for c in ["x", "y", "z"]]
            np.savetxt(subject_dir / "skeleton.csv", flat_skeleton, delimiter=",",
                       header=",".join(cols), comments="")

            labels_rows.append(f"{sid},{class_idx},{class_names[class_idx]}")
            subject_id += 1

    # Labels file
    with open(output_dir / "labels.csv", "w") as f:
        f.write("subject_id,label,class_name\n")
        for row in labels_rows:
            f.write(row + "\n")

    print(f"샘플 데이터 생성 완료: {output_dir}/")
    print(f"  피험자 수: {subject_id - 1} ({num_subjects_per_class} x {num_classes} classes)")
    print(f"  클래스: {class_names}")
    print(f"\n실행 방법:")
    print(f"  python run_real_data.py --data-dir data/sample/ --format folder --label-file labels.csv")


if __name__ == "__main__":
    main()
