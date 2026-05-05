"""ML 튜닝 + DL 파인튜닝 파이프라인 테스트."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.ml_tuner import MLClassifierTuner, TuningResult
from src.training.finetune import GaitModelFineTuner, FinetunePhase, FinetuneResult
from src.models.multimodal_gait_net import MultimodalGaitNet


# ── ML Tuner 테스트 ────────────────────────────────────────────────

class TestMLClassifierTuner:
    @pytest.fixture
    def tuner(self):
        return MLClassifierTuner(cv_folds=3)

    def test_generate_data(self, tuner):
        X, y = tuner._generate_data(n_per_class=20)
        assert X.shape == (220, 13)
        assert len(np.unique(y)) == 11

    def test_run_minimal(self, tuner):
        result = tuner.run(n_trials=3)
        assert isinstance(result, TuningResult)
        assert result.best_f1 > 0
        assert result.best_accuracy > 0
        assert "rf_n_estimators" in result.best_params
        assert "gb_learning_rate" in result.best_params
        assert len(result.feature_importance) == 13

    def test_build_classifier(self, tuner):
        result = tuner.run(n_trials=2)
        clf = result.build_classifier()
        assert clf.is_trained

        features = {
            "gait_speed": 1.2, "cadence": 115, "stride_regularity": 0.85,
            "step_symmetry": 0.92, "cop_sway": 0.04, "ml_variability": 0.06,
            "heel_pressure_ratio": 0.32, "forefoot_pressure_ratio": 0.45,
            "arch_index": 0.25, "pressure_asymmetry": 0.05,
            "acceleration_rms": 1.5, "acceleration_variability": 0.15,
            "trunk_sway": 2.0,
        }
        prediction = clf.predict(features)
        assert prediction.predicted_class is not None
        assert prediction.confidence > 0

    def test_save_results(self, tuner, tmp_path):
        result = tuner.run(n_trials=2, output_dir=str(tmp_path / "ml_results"))
        assert (tmp_path / "ml_results" / "best_params.json").exists()


# ── DL Fine-tuner 테스트 ───────────────────────────────────────────

def _make_config():
    return {
        "data": {
            "sequence_length": 32,
            "imu_channels": 6,
            "pressure_grid_size": [16, 8],
            "skeleton_joints": 17,
            "skeleton_dims": 3,
            "num_classes": 4,
        },
        "model": {
            "imu_encoder": {
                "conv_channels": [16, 32],
                "kernel_size": 3,
                "lstm_layers": 1,
                "dropout": 0.1,
            },
            "pressure_encoder": {
                "conv_channels": [8, 16],
                "kernel_size": 3,
                "dropout": 0.1,
            },
            "skeleton_encoder": {
                "gcn_channels": [32, 64],
                "temporal_kernel": 3,
                "dropout": 0.1,
            },
            "fusion": {
                "embed_dim": 64,
                "num_heads": 4,
                "ff_dim": 128,
                "num_layers": 1,
                "dropout": 0.1,
            },
            "classifier": {
                "hidden_dims": [32],
                "dropout": 0.1,
            },
        },
    }


def _make_loader(config, n=32):
    seq_len = config["data"]["sequence_length"]
    imu = torch.randn(n, 6, seq_len)
    pressure = torch.randn(n, seq_len, 1, 16, 8)
    skeleton = torch.randn(n, 3, seq_len, 17)
    labels = torch.randint(0, config["data"]["num_classes"], (n,))

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, imu, pressure, skeleton, labels):
            self.imu = imu
            self.pressure = pressure
            self.skeleton = skeleton
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "imu": self.imu[idx],
                "pressure": self.pressure[idx],
                "skeleton": self.skeleton[idx],
                "label": self.labels[idx],
            }

    ds = DictDataset(imu, pressure, skeleton, labels)
    return DataLoader(ds, batch_size=8, shuffle=True)


class TestGaitModelFineTuner:
    @pytest.fixture
    def config(self):
        return _make_config()

    @pytest.fixture
    def loaders(self, config):
        train = _make_loader(config, n=32)
        val = _make_loader(config, n=16)
        test = _make_loader(config, n=16)
        return train, val, test

    def test_freeze_unfreeze(self, config):
        ft = GaitModelFineTuner(config)
        ft._freeze_all()
        assert ft.model.get_num_trainable_params() == 0

        ft._unfreeze_modules(["classifier"])
        trainable = ft.model.get_num_trainable_params()
        total = ft.model.get_num_params()
        assert 0 < trainable < total

    def test_progressive_unfreeze_increases_params(self, config):
        ft = GaitModelFineTuner(config)
        counts = []
        for phase in ft.phases:
            ft._freeze_all()
            ft._unfreeze_modules(phase.unfreeze)
            counts.append(ft.model.get_num_trainable_params())
        assert counts[0] < counts[1] < counts[2]

    def test_discriminative_lr(self, config):
        ft = GaitModelFineTuner(config)
        phase = FinetunePhase(
            name="full",
            epochs=1,
            lr=1e-3,
            unfreeze=["imu_encoder", "fusion", "classifier"],
        )
        ft._freeze_all()
        ft._unfreeze_modules(phase.unfreeze)
        optimizer = ft._build_optimizer(phase)
        lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
        assert lrs["imu_encoder"] < lrs["fusion"] < lrs["classifier"]

    def test_short_finetune(self, config, loaders):
        train_loader, val_loader, test_loader = loaders
        phases = [
            FinetunePhase(name="head", epochs=2, lr=1e-3, unfreeze=["classifier"]),
            FinetunePhase(name="full", epochs=2, lr=1e-4,
                         unfreeze=["imu_encoder", "pressure_encoder",
                                   "skeleton_encoder", "fusion", "classifier"]),
        ]
        ft = GaitModelFineTuner(config, phases=phases)
        result = ft.run(train_loader, val_loader, test_loader)

        assert isinstance(result, FinetuneResult)
        assert result.phases_completed == 2
        assert result.total_epochs >= 2
        assert result.best_val_accuracy >= 0
        assert result.final_test_metrics is not None
        assert len(result.history["train_loss"]) == result.total_epochs

    def test_save_checkpoint(self, config, loaders, tmp_path):
        train_loader, val_loader, _ = loaders
        phases = [
            FinetunePhase(name="head", epochs=1, lr=1e-3, unfreeze=["classifier"]),
        ]
        ft = GaitModelFineTuner(config, phases=phases)
        result = ft.run(train_loader, val_loader)
        ft.save(result, str(tmp_path / "ft_out"))
        assert (tmp_path / "ft_out" / "finetuned_model.pt").exists()

        checkpoint = torch.load(tmp_path / "ft_out" / "finetuned_model.pt", weights_only=True)
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint

    def test_load_pretrained_and_finetune(self, config, loaders, tmp_path):
        train_loader, val_loader, _ = loaders

        pretrained = MultimodalGaitNet(config)
        torch.save(
            {"model_state_dict": pretrained.state_dict()},
            tmp_path / "pretrained.pt",
        )

        phases = [
            FinetunePhase(name="head", epochs=1, lr=1e-3, unfreeze=["classifier"]),
        ]
        ft = GaitModelFineTuner(
            config,
            pretrained_path=str(tmp_path / "pretrained.pt"),
            phases=phases,
        )
        result = ft.run(train_loader, val_loader)
        assert result.phases_completed == 1
