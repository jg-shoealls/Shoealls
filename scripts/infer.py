"""Run inference on a single synthetic sample using a saved checkpoint."""

import sys
import torch
import numpy as np

from src.data.synthetic import generate_synthetic_dataset, CLASS_NAMES
from src.data.dataset import MultimodalGaitDataset
from src.models.multimodal_gait_net import MultimodalGaitNet


def infer(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    num_classes = config["data"]["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalGaitNet(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Generate one sample per class for demonstration
    data_cfg = config["data"]
    dataset_dict = generate_synthetic_dataset(
        num_samples_per_class=1,
        num_classes=num_classes,
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

    names = CLASS_NAMES[:num_classes]
    print(f"\nInference with {checkpoint_path}")
    print(f"Device: {device}  |  Classes: {num_classes}")
    print("-" * 55)
    print(f"{'Sample':<12} {'True':<25} {'Predicted':<25} {'Conf':>6}")
    print("-" * 55)

    correct = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            label = sample.pop("label").item()
            batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[0]
            pred = probs.argmax().item()
            conf = probs[pred].item()

            correct += pred == label
            mark = "O" if pred == label else "X"
            print(f"{mark} Sample {i:<4} {names[label]:<25} {names[pred]:<25} {conf:>5.1%}")

    print("-" * 55)
    print(f"Accuracy: {correct}/{len(dataset)}  ({correct/len(dataset):.1%})")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/run_011/best_model.pt"
    infer(path)
