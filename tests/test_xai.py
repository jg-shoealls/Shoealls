"""Tests for XAI visualization module."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.reasoning_engine import (
    GaitReasoningEngine,
    CrossModalEvidenceCollector,
)
from src.validation.xai_visualize import (
    plot_cross_modal_attention,
    compute_pressure_gradcam,
    plot_pressure_gradcam,
    compute_zone_importance,
    plot_modality_contribution,
    plot_xai_dashboard,
)


def load_config():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


def make_batch(batch_size=2):
    return {
        "imu": torch.randn(batch_size, 6, 128),
        "pressure": torch.randn(batch_size, 128, 1, 16, 8).abs(),
        "skeleton": torch.randn(batch_size, 3, 128, 17),
    }


class TestCrossModalAttentionVisualization:
    def test_evidence_collector_returns_attn_weights(self):
        module = CrossModalEvidenceCollector(embed_dim=128, num_heads=4)
        features = [torch.randn(2, 10, 128) for _ in range(3)]
        deviations = [torch.randn(2, 128) for _ in range(3)]

        out = module(features, deviations)
        assert "cross_attn_weights" in out
        # (B, num_heads, 3, 3)
        assert out["cross_attn_weights"].shape == (2, 4, 3, 3)

    def test_plot_cross_modal_attention(self):
        weights = np.random.rand(2, 4, 3, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attn.png"
            plot_cross_modal_attention(weights, path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_reasoning_engine_returns_attn(self):
        config = load_config()
        engine = GaitReasoningEngine(config)
        batch = make_batch(1)

        result = engine.reason(batch)
        assert "cross_attn_weights" in result["evidence"]
        attn = result["evidence"]["cross_attn_weights"]
        assert attn.shape[2] == 3
        assert attn.shape[3] == 3


class TestGradCAM:
    def test_compute_gradcam_shape(self):
        config = load_config()
        engine = GaitReasoningEngine(config)
        batch = make_batch(1)

        gradcam = compute_pressure_gradcam(engine, batch, target_class=0)
        B, T = batch["pressure"].shape[:2]
        H, W = batch["pressure"].shape[3], batch["pressure"].shape[4]
        assert gradcam.shape == (B, T, H, W)

    def test_gradcam_values_normalized(self):
        config = load_config()
        engine = GaitReasoningEngine(config)
        batch = make_batch(1)

        gradcam = compute_pressure_gradcam(engine, batch, target_class=0)
        assert gradcam.min() >= 0.0
        assert gradcam.max() <= 1.0 + 1e-6

    def test_gradcam_auto_target(self):
        """Test Grad-CAM with automatic target class selection."""
        config = load_config()
        engine = GaitReasoningEngine(config)
        batch = make_batch(1)

        gradcam = compute_pressure_gradcam(engine, batch, target_class=None)
        assert gradcam.shape[0] == 1
        assert gradcam.shape[2] == 16
        assert gradcam.shape[3] == 8

    def test_plot_pressure_gradcam(self):
        gradcam = np.random.rand(1, 128, 16, 8).astype(np.float32)
        pressure = np.random.rand(1, 128, 1, 16, 8).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gradcam.png"
            plot_pressure_gradcam(gradcam, pressure, path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_compute_zone_importance(self):
        gradcam = np.random.rand(1, 128, 16, 8).astype(np.float32)
        zones = compute_zone_importance(gradcam, sample_idx=0)

        assert len(zones) == 7
        for name, info in zones.items():
            assert "importance" in info
            assert "peak" in info
            assert "label" in info
            assert 0.0 <= info["importance"] <= 1.0


class TestModalityContribution:
    def test_plot_modality_contribution(self):
        weights = np.array([[0.4, 0.35, 0.25]])
        support = np.array([[0.6, 0.5, 0.4]])

        anomaly_results = []
        for _ in range(3):
            anomaly_results.append({
                "anomaly_scores": torch.rand(1, 8),
                "temporal_heatmap": torch.rand(1, 16),
                "deviation": torch.randn(1, 128),
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "contribution.png"
            plot_modality_contribution(weights, support, anomaly_results, path)
            assert path.exists()
            assert path.stat().st_size > 0


class TestXAIDashboard:
    def test_full_dashboard(self):
        config = load_config()
        engine = GaitReasoningEngine(config)
        batch = make_batch(1)
        result = engine.reason(batch)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dashboard.png"
            plot_xai_dashboard(result, batch, engine, path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_encoder_features_in_result(self):
        config = load_config()
        engine = GaitReasoningEngine(config)
        batch = make_batch(1)
        result = engine.reason(batch)

        assert "encoder_features" in result
        assert "imu" in result["encoder_features"]
        assert "pressure" in result["encoder_features"]
        assert "skeleton" in result["encoder_features"]
