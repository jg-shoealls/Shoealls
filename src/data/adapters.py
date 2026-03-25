"""실제 데이터 연동 어댑터.

지원 데이터 형식:
  1. CSV/Excel: 센서별 CSV 파일 (가장 흔한 형태)
  2. NumPy: .npy/.npz 파일
  3. 폴더 구조: 피험자별 폴더 안에 센서 파일
  4. HDF5: 대규모 데이터셋

사용법:
    # 1) 폴더 구조에서 자동 로드
    adapter = FolderDataAdapter("data/collected/")
    dataset = adapter.to_dataset(sequence_length=128)

    # 2) CSV 파일에서 로드
    adapter = CSVDataAdapter(
        imu_pattern="data/imu_*.csv",
        pressure_pattern="data/pressure_*.csv",
        skeleton_pattern="data/skeleton_*.csv",
        label_file="data/labels.csv",
    )
    dataset = adapter.to_dataset()
"""

from pathlib import Path
from typing import Optional

import numpy as np


class FolderDataAdapter:
    """폴더 구조 기반 데이터 로더.

    기대하는 폴더 구조:
        data_root/
        ├── subject_001/
        │   ├── imu.csv           # (T, 6) 또는 (T, 7) 첫 열=시간
        │   ├── pressure.csv      # (T, 128) 또는 (T, H*W)
        │   ├── skeleton.csv      # (T, 51) = 17 joints × 3 coords
        │   └── meta.json         # {"label": 0, "disease": 2, ...} (선택)
        ├── subject_002/
        │   └── ...
        └── labels.csv            # subject_id, gait_label, disease_label, ...
    """

    def __init__(
        self,
        data_root: str,
        imu_filename: str = "imu.csv",
        pressure_filename: str = "pressure.csv",
        skeleton_filename: str = "skeleton.csv",
        label_file: Optional[str] = None,
        imu_cols: Optional[list] = None,
        pressure_grid_size: tuple = (16, 8),
        num_joints: int = 17,
        delimiter: str = ",",
        has_header: bool = True,
        has_timestamp_col: bool = False,
    ):
        self.data_root = Path(data_root)
        self.imu_filename = imu_filename
        self.pressure_filename = pressure_filename
        self.skeleton_filename = skeleton_filename
        self.pressure_grid_size = pressure_grid_size
        self.num_joints = num_joints
        self.delimiter = delimiter
        self.has_header = has_header
        self.has_timestamp_col = has_timestamp_col
        self.imu_cols = imu_cols  # e.g., [1,2,3,4,5,6] to select specific columns

        # Load labels
        self.labels_df = None
        if label_file:
            self._load_labels(label_file)

        # Discover subjects
        self.subject_dirs = sorted([
            d for d in self.data_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not self.subject_dirs:
            raise FileNotFoundError(
                f"No subject directories found in {data_root}\n"
                f"Expected structure:\n"
                f"  {data_root}/subject_001/imu.csv\n"
                f"  {data_root}/subject_001/pressure.csv\n"
                f"  {data_root}/subject_001/skeleton.csv"
            )

        print(f"Found {len(self.subject_dirs)} subjects in {data_root}")

    def _load_labels(self, label_file: str):
        """Load label CSV file."""
        import pandas as pd
        path = self.data_root / label_file if not Path(label_file).is_absolute() else Path(label_file)
        self.labels_df = pd.read_csv(path)

    def _load_csv(self, filepath: Path) -> np.ndarray:
        """Load a CSV file into numpy array."""
        skip = 1 if self.has_header else 0
        data = np.loadtxt(filepath, delimiter=self.delimiter, skiprows=skip)
        if self.has_timestamp_col:
            data = data[:, 1:]  # Remove timestamp column
        return data.astype(np.float32)

    def _load_subject(self, subject_dir: Path) -> dict:
        """Load all modality data for one subject."""
        result = {}

        # IMU
        imu_path = subject_dir / self.imu_filename
        if imu_path.exists():
            imu = self._load_csv(imu_path)
            if self.imu_cols:
                imu = imu[:, self.imu_cols]
            if imu.shape[1] != 6:
                raise ValueError(
                    f"IMU data should have 6 columns (ax,ay,az,gx,gy,gz), "
                    f"got {imu.shape[1]} in {imu_path}\n"
                    f"Use imu_cols parameter to select the right columns."
                )
            result["imu"] = imu
        else:
            raise FileNotFoundError(f"IMU file not found: {imu_path}")

        # Pressure
        pressure_path = subject_dir / self.pressure_filename
        if pressure_path.exists():
            pressure = self._load_csv(pressure_path)
            h, w = self.pressure_grid_size
            if pressure.shape[1] == h * w:
                pressure = pressure.reshape(-1, h, w)
            elif pressure.ndim == 2 and pressure.shape[1] != h * w:
                raise ValueError(
                    f"Pressure data has {pressure.shape[1]} columns, "
                    f"expected {h}x{w}={h*w} for grid_size={self.pressure_grid_size}\n"
                    f"Adjust pressure_grid_size parameter."
                )
            result["pressure"] = pressure
        else:
            raise FileNotFoundError(f"Pressure file not found: {pressure_path}")

        # Skeleton
        skeleton_path = subject_dir / self.skeleton_filename
        if skeleton_path.exists():
            skeleton = self._load_csv(skeleton_path)
            j = self.num_joints
            if skeleton.shape[1] == j * 3:
                skeleton = skeleton.reshape(-1, j, 3)
            elif skeleton.shape[1] == j * 2:
                # 2D skeleton → pad z=0
                skeleton_2d = skeleton.reshape(-1, j, 2)
                skeleton = np.zeros((*skeleton_2d.shape[:2], 3), dtype=np.float32)
                skeleton[:, :, :2] = skeleton_2d
            else:
                raise ValueError(
                    f"Skeleton data has {skeleton.shape[1]} columns, "
                    f"expected {j}*3={j*3} or {j}*2={j*2}\n"
                    f"Adjust num_joints parameter."
                )
            result["skeleton"] = skeleton
        else:
            raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")

        return result

    def load_all(self) -> dict:
        """Load all subjects into a dataset dictionary.

        Returns:
            Dict with 'imu', 'pressure', 'skeleton' (lists of arrays),
            'labels' (np.ndarray), 'subject_ids' (list of str).
        """
        imu_list, pressure_list, skeleton_list = [], [], []
        labels, subject_ids = [], []
        skipped = []

        for subject_dir in self.subject_dirs:
            try:
                data = self._load_subject(subject_dir)
                imu_list.append(data["imu"])
                pressure_list.append(data["pressure"])
                skeleton_list.append(data["skeleton"])
                subject_ids.append(subject_dir.name)

                # Get label
                if self.labels_df is not None:
                    row = self.labels_df[
                        self.labels_df.iloc[:, 0].astype(str) == subject_dir.name
                    ]
                    if len(row) > 0:
                        labels.append(int(row.iloc[0, 1]))
                    else:
                        labels.append(-1)  # Unknown label
                else:
                    labels.append(-1)

            except (FileNotFoundError, ValueError) as e:
                skipped.append((subject_dir.name, str(e)))

        if skipped:
            print(f"\nSkipped {len(skipped)} subjects:")
            for name, reason in skipped[:5]:
                print(f"  {name}: {reason}")
            if len(skipped) > 5:
                print(f"  ... and {len(skipped) - 5} more")

        print(f"Successfully loaded: {len(imu_list)} subjects")

        return {
            "imu": imu_list,
            "pressure": pressure_list,
            "skeleton": skeleton_list,
            "labels": np.array(labels, dtype=np.int64),
            "subject_ids": subject_ids,
        }

    def to_dataset(self, sequence_length: int = 128):
        """Load data and create a PyTorch dataset."""
        from .dataset import MultimodalGaitDataset

        data_dict = self.load_all()
        return MultimodalGaitDataset(
            data_dict,
            sequence_length=sequence_length,
            grid_size=self.pressure_grid_size,
            num_joints=self.num_joints,
        )


class CSVDataAdapter:
    """개별 CSV 파일 기반 데이터 로더.

    각 피험자/시행의 센서 데이터가 별도 CSV 파일인 경우.

    파일 이름 규칙 예시:
        imu_001.csv, imu_002.csv, ...
        pressure_001.csv, pressure_002.csv, ...
        skeleton_001.csv, skeleton_002.csv, ...
    """

    def __init__(
        self,
        imu_files: list[str],
        pressure_files: list[str],
        skeleton_files: list[str],
        labels: list[int],
        pressure_grid_size: tuple = (16, 8),
        num_joints: int = 17,
        delimiter: str = ",",
        has_header: bool = True,
    ):
        assert len(imu_files) == len(pressure_files) == len(skeleton_files) == len(labels), \
            "All file lists and labels must have the same length"

        self.imu_files = [Path(f) for f in imu_files]
        self.pressure_files = [Path(f) for f in pressure_files]
        self.skeleton_files = [Path(f) for f in skeleton_files]
        self.labels = labels
        self.pressure_grid_size = pressure_grid_size
        self.num_joints = num_joints
        self.delimiter = delimiter
        self.has_header = has_header

    def load_all(self) -> dict:
        """Load all files into dataset dictionary."""
        skip = 1 if self.has_header else 0
        h, w = self.pressure_grid_size
        j = self.num_joints

        imu_list, pressure_list, skeleton_list = [], [], []

        for i, (imu_f, pres_f, skel_f) in enumerate(
            zip(self.imu_files, self.pressure_files, self.skeleton_files)
        ):
            imu = np.loadtxt(imu_f, delimiter=self.delimiter, skiprows=skip).astype(np.float32)
            pressure = np.loadtxt(pres_f, delimiter=self.delimiter, skiprows=skip).astype(np.float32)
            skeleton = np.loadtxt(skel_f, delimiter=self.delimiter, skiprows=skip).astype(np.float32)

            pressure = pressure.reshape(-1, h, w)
            skeleton = skeleton.reshape(-1, j, 3)

            imu_list.append(imu)
            pressure_list.append(pressure)
            skeleton_list.append(skeleton)

        return {
            "imu": imu_list,
            "pressure": pressure_list,
            "skeleton": skeleton_list,
            "labels": np.array(self.labels, dtype=np.int64),
        }

    def to_dataset(self, sequence_length: int = 128):
        from .dataset import MultimodalGaitDataset

        data_dict = self.load_all()
        return MultimodalGaitDataset(
            data_dict,
            sequence_length=sequence_length,
            grid_size=self.pressure_grid_size,
            num_joints=self.num_joints,
        )


class NumpyDataAdapter:
    """NumPy .npy/.npz 파일 기반 데이터 로더.

    단일 .npz 파일에 모든 데이터가 저장된 경우:
        np.savez("gait_data.npz",
            imu=imu_array,           # (N, T, 6)
            pressure=pressure_array, # (N, T, H, W)
            skeleton=skeleton_array, # (N, T, J, 3)
            labels=labels_array,     # (N,)
        )
    """

    def __init__(self, npz_path: str):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"File not found: {npz_path}")

    def load_all(self) -> dict:
        data = np.load(self.npz_path, allow_pickle=True)

        required = ["imu", "pressure", "skeleton", "labels"]
        for key in required:
            if key not in data:
                raise KeyError(
                    f"Missing key '{key}' in {self.npz_path}\n"
                    f"Expected keys: {required}\n"
                    f"Found keys: {list(data.keys())}"
                )

        # Convert from (N, T, ...) arrays to list of individual arrays
        imu_list = [data["imu"][i] for i in range(len(data["imu"]))]
        pressure_list = [data["pressure"][i] for i in range(len(data["pressure"]))]
        skeleton_list = [data["skeleton"][i] for i in range(len(data["skeleton"]))]

        print(f"Loaded {len(imu_list)} samples from {self.npz_path}")
        print(f"  IMU shape per sample: {imu_list[0].shape}")
        print(f"  Pressure shape per sample: {pressure_list[0].shape}")
        print(f"  Skeleton shape per sample: {skeleton_list[0].shape}")

        return {
            "imu": imu_list,
            "pressure": pressure_list,
            "skeleton": skeleton_list,
            "labels": data["labels"].astype(np.int64),
        }

    def to_dataset(self, sequence_length: int = 128, grid_size=(16, 8), num_joints=17):
        from .dataset import MultimodalGaitDataset

        data_dict = self.load_all()
        return MultimodalGaitDataset(
            data_dict,
            sequence_length=sequence_length,
            grid_size=grid_size,
            num_joints=num_joints,
        )
