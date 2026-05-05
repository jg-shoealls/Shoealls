"""사전학습 인코더 및 HybridGaitNet 테스트."""

import pytest
import torch

from src.models.pretrained_encoders import (
    LIMUBERTEncoder,
    MobileNetV2PressureEncoder,
    CTRGCNEncoder,
)
from src.models.hybrid_gait_net import HybridGaitNet


B, T, J = 2, 32, 17


def _config(num_classes=4):
    return {
        "data": {
            "num_classes": num_classes,
            "skeleton_joints": J,
            "imu_channels": 6,
        },
        "model": {
            "fusion": {"embed_dim": 64, "num_heads": 4, "num_layers": 1, "ff_dim": 128, "dropout": 0.1},
            "classifier": {"hidden_dims": [32], "dropout": 0.1},
        },
    }


# ── LIMUBERTEncoder ───────────────────────────────────────────────

class TestLIMUBERTEncoder:
    @pytest.fixture
    def enc(self):
        return LIMUBERTEncoder(embed_dim=32, num_heads=2, num_layers=2,
                               patch_size=8, embed_dim_out=64)

    def test_output_shape(self, enc):
        x = torch.randn(B, 6, T)
        out = enc(x)
        assert out.shape == (B, T, 64)

    def test_output_shape_odd_T(self, enc):
        x = torch.randn(B, 6, 33)
        out = enc(x)
        assert out.shape == (B, 33, 64)

    def test_param_count(self):
        enc = LIMUBERTEncoder(embed_dim=72, num_heads=4, num_layers=4,
                              patch_size=8, embed_dim_out=72)
        n = sum(p.numel() for p in enc.parameters())
        assert n < 400_000

    def test_same_embed_dim_no_proj(self):
        enc = LIMUBERTEncoder(embed_dim=64, num_heads=4, num_layers=2,
                              patch_size=8, embed_dim_out=64)
        x = torch.randn(B, 6, T)
        out = enc(x)
        assert out.shape == (B, T, 64)


# ── MobileNetV2PressureEncoder ────────────────────────────────────

class TestMobileNetV2PressureEncoder:
    @pytest.fixture
    def enc(self):
        return MobileNetV2PressureEncoder(embed_dim=64, dropout=0.0, pretrained=False)

    def test_output_shape_small_grid(self, enc):
        x = torch.randn(B, T, 1, 16, 8)
        out = enc(x)
        assert out.shape == (B, T, 64)

    def test_output_shape_larger_grid(self, enc):
        x = torch.randn(B, T, 1, 64, 32)
        out = enc(x)
        assert out.shape == (B, T, 64)

    def test_single_channel_input(self, enc):
        x = torch.randn(1, 1, 1, 16, 8)
        out = enc(x)
        assert out.shape == (1, 1, 64)

    def test_param_count(self):
        enc = MobileNetV2PressureEncoder(pretrained=False, width_mult=0.35)
        n = sum(p.numel() for p in enc.parameters())
        assert n < 1_000_000


# ── CTRGCNEncoder ─────────────────────────────────────────────────

class TestCTRGCNEncoder:
    @pytest.fixture
    def enc(self):
        return CTRGCNEncoder(in_channels=3, num_joints=J,
                             gcn_channels=[32, 64], embed_dim=64, dropout=0.0)

    def test_output_shape(self, enc):
        x = torch.randn(B, 3, T, J)
        out = enc(x)
        assert out.shape == (B, T, 64)

    def test_adj_is_normalized(self, enc):
        adj = enc.adj
        assert adj.shape == (J, J)
        assert adj.max() <= 1.0 + 1e-5
        assert adj.min() >= 0.0

    def test_dynamic_alpha_learnable(self, enc):
        has_alpha = any("alpha" in n for n, _ in enc.named_parameters())
        assert has_alpha

    def test_param_count(self):
        enc = CTRGCNEncoder(gcn_channels=[64, 128], num_joints=17, embed_dim=128)
        n = sum(p.numel() for p in enc.parameters())
        assert n < 3_000_000


# ── HybridGaitNet ─────────────────────────────────────────────────

class TestHybridGaitNet:
    @pytest.fixture
    def model(self):
        return HybridGaitNet(
            num_classes=4, embed_dim=64,
            imu_embed_dim=32, imu_heads=2, imu_layers=2, imu_patch_size=8,
            skeleton_joints=J, skeleton_gcn_channels=[32, 64],
            fusion_heads=4, fusion_layers=1, fusion_ff_dim=128,
            classifier_hidden=[32], pressure_pretrained=False,
        )

    @pytest.fixture
    def batch(self):
        return {
            "imu": torch.randn(B, 6, T),
            "pressure": torch.randn(B, T, 1, 16, 8),
            "skeleton": torch.randn(B, 3, T, J),
        }

    def test_forward_shape(self, model, batch):
        logits = model(batch)
        assert logits.shape == (B, 4)

    def test_from_config(self, batch):
        config = _config()
        model = HybridGaitNet.from_config(config, pretrained=False)
        logits = model(batch)
        assert logits.shape == (B, 4)

    def test_freeze_encoders(self, model):
        model.freeze_encoders()
        enc_params = sum(
            p.numel() for enc in [model.imu_encoder, model.pressure_encoder, model.skeleton_encoder]
            for p in enc.parameters()
        )
        trainable_enc = sum(
            p.numel() for enc in [model.imu_encoder, model.pressure_encoder, model.skeleton_encoder]
            for p in enc.parameters() if p.requires_grad
        )
        assert trainable_enc == 0
        # 퓨전 + 분류기는 학습 가능
        trainable_total = model.get_num_trainable_params()
        assert trainable_total < enc_params

    def test_progressive_unfreeze(self, model):
        model.freeze_all()
        p0 = model.get_num_trainable_params()
        assert p0 == 0

        model.unfreeze_fusion()
        p1 = model.get_num_trainable_params()

        model.freeze_encoders()   # encoders still frozen; unfreeze classifier
        model.unfreeze_all()
        p2 = model.get_num_trainable_params()

        assert 0 == p0 < p1 < p2 == model.get_num_params()

    def test_param_summary(self, model):
        summary = model.param_summary()
        assert "LIMU-BERT" in summary
        assert "MobileNetV2" in summary
        assert "CTR-GCN" in summary
        assert "합계" in summary

    def test_gradient_flows(self, model, batch):
        model.unfreeze_all()
        logits = model(batch)
        loss = logits.sum()
        loss.backward()
        # 분류기 최소 하나 이상에 gradient 존재
        assert any(
            p.grad is not None
            for p in model.classifier.parameters()
        )

    def test_num_classes(self):
        for nc in [4, 11]:
            model = HybridGaitNet(
                num_classes=nc, embed_dim=64,
                imu_embed_dim=32, imu_heads=2, imu_layers=2,
                skeleton_joints=J, skeleton_gcn_channels=[32, 64],
                fusion_heads=4, fusion_layers=1, fusion_ff_dim=128,
                classifier_hidden=[32], pressure_pretrained=False,
            )
            batch = {
                "imu": torch.randn(1, 6, T),
                "pressure": torch.randn(1, T, 1, 16, 8),
                "skeleton": torch.randn(1, 3, T, J),
            }
            assert model(batch).shape == (1, nc)
