"""WearGait-PD Dataset Adapter.

WearGait-PD CSV 파일(SelfPace.csv 등)을 로드하여
MultimodalGaitNet에서 사용할 수 있는 형식으로 변환합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from .dataset import MultimodalGaitDataset

class WearGaitPDAdapter:
    """WearGait-PD CSV 데이터 로더."""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.hc_dir = self.data_root / "HC"
        self.pd_dir = self.data_root / "PD"
        
        if not self.hc_dir.exists() or not self.pd_dir.exists():
            raise FileNotFoundError(f"HC or PD directory not found in {data_root}")

    def load_all(self, task: str = "SelfPace"):
        """HC와 PD 폴더에서 특정 태스크의 CSV를 모두 로드."""
        imu_list, pressure_list, skeleton_list = [], [], []
        labels = []
        subject_ids = []

        # HC (Normal) -> Label 0
        hc_files = list(self.hc_dir.glob(f"*{task}.csv"))
        print(f"Loading {len(hc_files)} HC files...")
        for f in hc_files:
            data = self._parse_csv(f)
            if data:
                imu_list.append(data["imu"])
                pressure_list.append(data["pressure"])
                skeleton_list.append(data["skeleton"])
                labels.append(0)
                subject_ids.append(f.stem)

        # PD (Parkinsonian) -> Label 3
        pd_files = list(self.pd_dir.glob(f"*{task}.csv"))
        print(f"Loading {len(pd_files)} PD files...")
        for f in pd_files:
            data = self._parse_csv(f)
            if data:
                imu_list.append(data["imu"])
                pressure_list.append(data["pressure"])
                skeleton_list.append(data["skeleton"])
                labels.append(3)
                subject_ids.append(f.stem)

        return {
            "imu": imu_list,
            "pressure": pressure_list,
            "skeleton": skeleton_list,
            "labels": np.array(labels, dtype=np.int64),
            "subject_ids": subject_ids
        }

    def _parse_csv(self, filepath: Path) -> dict | None:
        """단일 CSV 파일을 파싱하여 모달리티별 배열 추출."""
        try:
            # 첫 1줄(헤더)을 제외하고 로드
            df = pd.read_csv(filepath, low_memory=False)
            
            # 1. IMU (LowerBack)
            # WearGait-PD columns: LowerBack_Acc_X/Y/Z, LowerBack_Gyr_X/Y/Z
            imu_cols = [
                'LowerBack_Acc_X', 'LowerBack_Acc_Y', 'LowerBack_Acc_Z',
                'LowerBack_Gyr_X', 'LowerBack_Gyr_Y', 'LowerBack_Gyr_Z'
            ]
            if not all(c in df.columns for c in imu_cols):
                return None
            imu = df[imu_cols].values.astype(np.float32)

            # 2. Pressure (Simplified)
            # WearGait-PD has 'L Foot Pressure' and 'R Foot Pressure'
            # We map these to a dummy 16x8 grid (128 values)
            # Here we just put L pressure in the left half and R in the right half
            pres_l = df['L Foot Pressure'].values.astype(np.float32)
            pres_r = df['R Foot Pressure'].values.astype(np.float32)
            
            # Create (T, 16, 8) dummy grid
            T = len(df)
            pressure = np.zeros((T, 16, 8), dtype=np.float32)
            pressure[:, :8, :] = pres_l[:, np.newaxis, np.newaxis] / 8.0 # Simplified distribution
            pressure[:, 8:, :] = pres_r[:, np.newaxis, np.newaxis] / 8.0
            pressure = pressure.reshape(T, 128)

            # 3. Skeleton (Dummy or Segment Orientation)
            # Use orientations of multiple segments to mimic skeleton info
            # Or just use zeros for now if we want to focus on IMU/Pressure
            skeleton = np.zeros((T, 17, 3), dtype=np.float32)
            
            return {
                "imu": imu,
                "pressure": pressure,
                "skeleton": skeleton
            }
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None

    def to_dataset(self, sequence_length: int = 128):
        """데이터를 로드하고 PyTorch 데이터셋 생성."""
        data_dict = self.load_all()
        return MultimodalGaitDataset(
            data_dict,
            sequence_length=sequence_length,
            grid_size=(16, 8),
            num_joints=17
        )
