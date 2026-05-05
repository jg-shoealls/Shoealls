"""모델 평가 유틸 — run_pipeline.py / run_visualize.py 공통 로직."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalGaitDataset
from src.data.synthetic import generate_synthetic_dataset
from src.models.multimodal_gait_net import MultimodalGaitNet


def load_checkpoint(
    path: str | Path,
    config: dict,
    device: torch.device,
) -> tuple[MultimodalGaitNet, dict]:
    """체크포인트를 로드하고 eval 모드 모델을 반환한다."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MultimodalGaitNet(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _eval_dataset(config: dict) -> tuple[MultimodalGaitDataset, dict]:
    """평가용 합성 데이터셋을 생성한다 (seed=99, 30 samples/class)."""
    data_cfg = config["data"]
    dataset_dict = generate_synthetic_dataset(
        num_samples_per_class=30,
        num_classes=data_cfg["num_classes"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
        seed=99,
    )
    dataset = MultimodalGaitDataset(
        dataset_dict,
        sequence_length=data_cfg["sequence_length"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )
    return dataset, dataset_dict


def run_ablation(
    config: dict,
    device: torch.device,
    ckpt_path: str | Path = "outputs/best_model.pt",
) -> dict:
    """모달리티 제거 실험 (Ablation Study).

    Returns:
        {전략 이름: accuracy} 딕셔너리
    """
    model, _ = load_checkpoint(ckpt_path, config, device)
    dataset, _ = _eval_dataset(config)
    loader = DataLoader(dataset, batch_size=32)

    strategies = {
        "IMU only":          {"pressure": True, "skeleton": True},
        "Pressure only":     {"imu": True,      "skeleton": True},
        "Skeleton only":     {"imu": True,       "pressure": True},
        "IMU + Pressure":    {"skeleton": True},
        "IMU + Skeleton":    {"pressure": True},
        "Pressure + Skeleton": {"imu": True},
        "All (Fusion)":      {},
    }

    results = {}
    for name, mask in strategies.items():
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("label")
                for key in mask:
                    batch[key] = torch.zeros_like(batch[key])
                logits = model(batch)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        acc = float((preds == labels).mean())
        results[name] = acc
        print(f"  {name:25s}: {acc:.4f}")

    return results


def run_evaluation(
    config: dict,
    device: torch.device,
    ckpt_path: str | Path = "outputs/best_model.pt",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """전체 평가: 예측값 + 확률 반환.

    Returns:
        (y_true, y_pred, probs, class_names)
    """
    model, _ = load_checkpoint(ckpt_path, config, device)
    dataset, dataset_dict = _eval_dataset(config)
    loader = DataLoader(dataset, batch_size=32)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        dataset_dict["class_names"],
    )
