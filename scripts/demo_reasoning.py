"""추론 엔진 데모: 합성 데이터로 4단계 추론 과정을 시연."""

import torch
import yaml

from src.data.synthetic import generate_synthetic_dataset
from src.data.preprocessing import preprocess_imu, preprocess_pressure, preprocess_skeleton
from src.models.reasoning_engine import GaitReasoningEngine


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    engine = GaitReasoningEngine(config)
    engine.eval()

    # 각 클래스별 합성 데이터 1개씩
    data = generate_synthetic_dataset(num_samples_per_class=1, seed=42)
    class_names = data["class_names"]

    for i in range(len(data["labels"])):
        label = data["labels"][i]
        name = class_names[label]

        # 전처리
        imu = preprocess_imu(data["imu"][i], 128)
        pressure = preprocess_pressure(data["pressure"][i], 128, (16, 8))
        skeleton = preprocess_skeleton(data["skeleton"][i], 128, 17)

        batch = {
            "imu": torch.from_numpy(imu).unsqueeze(0),
            "pressure": torch.from_numpy(pressure).unsqueeze(0),
            "skeleton": torch.from_numpy(skeleton).unsqueeze(0),
        }

        result = engine.reason(batch)
        report = engine.explain(result, sample_idx=0)

        print(f"\n{'#' * 60}")
        print(f"  입력 데이터: {name} (Ground Truth: class {label})")
        print(f"{'#' * 60}")
        print(report)


if __name__ == "__main__":
    main()
