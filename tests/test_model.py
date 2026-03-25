"""Unit tests for the multimodal gait model."""

import numpy as np
import torch
import yaml

from src.data.preprocessing import preprocess_imu, preprocess_pressure, preprocess_skeleton
from src.data.synthetic import generate_synthetic_dataset
from src.data.dataset import MultimodalGaitDataset
from src.models.multimodal_gait_net import MultimodalGaitNet
from src.models.encoders import IMUEncoder, PressureEncoder, SkeletonEncoder
from src.models.fusion import CrossModalAttentionFusion


def load_config():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


class TestPreprocessing:
    def test_preprocess_imu(self):
        raw = np.random.randn(200, 6).astype(np.float32)
        result = preprocess_imu(raw, target_length=128)
        assert result.shape == (6, 128)
        assert result.dtype == np.float32

    def test_preprocess_pressure(self):
        raw = np.random.rand(200, 16, 8).astype(np.float32)
        result = preprocess_pressure(raw, target_length=128, grid_size=(16, 8))
        assert result.shape == (128, 1, 16, 8)
        assert result.dtype == np.float32

    def test_preprocess_skeleton(self):
        raw = np.random.randn(200, 17, 3).astype(np.float32)
        result = preprocess_skeleton(raw, target_length=128, num_joints=17)
        assert result.shape == (3, 128, 17)
        assert result.dtype == np.float32


class TestSyntheticData:
    def test_generate_dataset(self):
        data = generate_synthetic_dataset(
            num_samples_per_class=5, num_classes=4, seed=0
        )
        assert len(data["imu"]) == 20
        assert len(data["pressure"]) == 20
        assert len(data["skeleton"]) == 20
        assert data["labels"].shape == (20,)
        assert set(data["labels"]) == {0, 1, 2, 3}


class TestEncoders:
    def test_imu_encoder(self):
        encoder = IMUEncoder(in_channels=6, lstm_hidden=64)
        x = torch.randn(2, 6, 128)
        out = encoder(x)
        assert out.shape[0] == 2
        assert out.shape[2] == 64

    def test_pressure_encoder(self):
        encoder = PressureEncoder(embed_dim=64)
        x = torch.randn(2, 32, 1, 16, 8)
        out = encoder(x)
        assert out.shape == (2, 32, 64)

    def test_skeleton_encoder(self):
        encoder = SkeletonEncoder(num_joints=17, embed_dim=64)
        x = torch.randn(2, 3, 128, 17)
        out = encoder(x)
        assert out.shape[0] == 2
        assert out.shape[2] == 64


class TestFusion:
    def test_cross_modal_fusion(self):
        fusion = CrossModalAttentionFusion(
            embed_dim=64, num_heads=4, num_modalities=3
        )
        features = [
            torch.randn(2, 10, 64),
            torch.randn(2, 8, 64),
            torch.randn(2, 12, 64),
        ]
        out = fusion(features)
        assert out.shape == (2, 64)


class TestFullModel:
    def test_forward_pass(self):
        config = load_config()
        model = MultimodalGaitNet(config)

        batch = {
            "imu": torch.randn(2, 6, 128),
            "pressure": torch.randn(2, 128, 1, 16, 8),
            "skeleton": torch.randn(2, 3, 128, 17),
        }

        logits = model(batch)
        assert logits.shape == (2, 4)

    def test_parameter_count(self):
        config = load_config()
        model = MultimodalGaitNet(config)
        assert model.get_num_params() > 0
        assert model.get_num_trainable_params() == model.get_num_params()

    def test_dataset_to_model(self):
        """Integration test: synthetic data -> dataset -> model."""
        config = load_config()
        data = generate_synthetic_dataset(num_samples_per_class=2, num_classes=4)

        dataset = MultimodalGaitDataset(
            data,
            sequence_length=config["data"]["sequence_length"],
            grid_size=tuple(config["data"]["pressure_grid_size"]),
            num_joints=config["data"]["skeleton_joints"],
        )

        sample = dataset[0]
        batch = {k: v.unsqueeze(0) for k, v in sample.items() if k != "label"}

        model = MultimodalGaitNet(config)
        logits = model(batch)
        assert logits.shape == (1, 4)
