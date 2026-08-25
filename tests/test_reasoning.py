"""Tests for the reasoning engine."""

import torch
import yaml

from src.models.reasoning_engine import (
    AnomalyDetectionModule,
    CrossModalEvidenceCollector,
    DifferentialDiagnosisChain,
    ConfidenceCalibrator,
    GaitReasoningEngine,
)


def load_config():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


def make_batch(batch_size=2):
    return {
        "imu": torch.randn(batch_size, 6, 128),
        "pressure": torch.randn(batch_size, 128, 1, 16, 8),
        "skeleton": torch.randn(batch_size, 3, 128, 17),
    }


class TestAnomalyDetection:
    def test_output_shapes(self):
        module = AnomalyDetectionModule(embed_dim=128, num_anomaly_types=8)
        features = torch.randn(2, 16, 128)
        out = module(features)

        assert out["anomaly_scores"].shape == (2, 8)
        assert out["temporal_heatmap"].shape == (2, 16)
        assert out["deviation"].shape == (2, 128)

    def test_scores_bounded(self):
        module = AnomalyDetectionModule(embed_dim=128)
        features = torch.randn(2, 16, 128)
        out = module(features)

        assert (out["anomaly_scores"] >= 0).all()
        assert (out["anomaly_scores"] <= 1).all()
        assert (out["temporal_heatmap"] >= 0).all()
        assert (out["temporal_heatmap"] <= 1).all()


class TestCrossModalEvidence:
    def test_output_shapes(self):
        module = CrossModalEvidenceCollector(embed_dim=128)
        features = [torch.randn(2, t, 128) for t in [16, 64, 64]]
        deviations = [torch.randn(2, 128) for _ in range(3)]

        out = module(features, deviations)
        assert out["evidence_embedding"].shape == (2, 128)
        assert out["modality_weights"].shape == (2, 3)
        assert out["cross_support"].shape == (2, 3)

    def test_weights_sum_to_one(self):
        module = CrossModalEvidenceCollector(embed_dim=128)
        features = [torch.randn(2, 10, 128) for _ in range(3)]
        deviations = [torch.randn(2, 128) for _ in range(3)]

        out = module(features, deviations)
        sums = out["modality_weights"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)


class TestDifferentialDiagnosis:
    def test_output_shapes(self):
        module = DifferentialDiagnosisChain(embed_dim=128, num_classes=4, num_reasoning_steps=3)
        evidence = torch.randn(2, 128)
        context = torch.randn(2, 128)

        out = module(evidence, context)
        assert out["hypothesis_logits"].shape == (2, 4)
        assert len(out["reasoning_trace"]) == 4  # initial + 3 steps
        assert out["pro_scores"].shape == (2, 4)
        assert out["con_scores"].shape == (2, 4)


class TestConfidenceCalibrator:
    def test_output_shapes(self):
        module = ConfidenceCalibrator(num_classes=4, num_reasoning_steps=3)
        trace = [torch.randn(2, 4) for _ in range(4)]

        out = module(
            hypothesis_logits=torch.randn(2, 4),
            pro_scores=torch.rand(2, 4),
            con_scores=torch.rand(2, 4),
            evidence_strength=torch.rand(2),
            reasoning_trace=trace,
            cross_support_mean=torch.rand(2),
        )

        assert out["calibrated_logits"].shape == (2, 4)
        assert out["calibrated_probs"].shape == (2, 4)
        assert out["uncertainty"].shape == (2,)

    def test_probs_sum_to_one(self):
        module = ConfidenceCalibrator(num_classes=4, num_reasoning_steps=3)
        trace = [torch.randn(2, 4) for _ in range(4)]

        out = module(
            hypothesis_logits=torch.randn(2, 4),
            pro_scores=torch.rand(2, 4),
            con_scores=torch.rand(2, 4),
            evidence_strength=torch.rand(2),
            reasoning_trace=trace,
            cross_support_mean=torch.rand(2),
        )

        sums = out["calibrated_probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_uncertainty_bounded(self):
        module = ConfidenceCalibrator(num_classes=4, num_reasoning_steps=3)
        trace = [torch.randn(2, 4) for _ in range(4)]

        out = module(
            hypothesis_logits=torch.randn(2, 4),
            pro_scores=torch.rand(2, 4),
            con_scores=torch.rand(2, 4),
            evidence_strength=torch.rand(2),
            reasoning_trace=trace,
            cross_support_mean=torch.rand(2),
        )

        assert (out["uncertainty"] >= 0).all()
        assert (out["uncertainty"] <= 1).all()


class TestGaitReasoningEngine:
    def test_full_reasoning(self):
        config = load_config()
        engine = GaitReasoningEngine(config)
        batch = make_batch(2)

        result = engine.reason(batch)

        assert "calibrated_probs" in result
        assert "prediction" in result
        assert "uncertainty" in result
        assert "anomaly_results" in result
        assert "evidence" in result
        assert "diagnosis" in result

        assert result["prediction"].shape == (2,)
        assert result["calibrated_probs"].shape == (2, 4)

    def test_explain_output(self):
        config = load_config()
        engine = GaitReasoningEngine(config)
        batch = make_batch(1)

        result = engine.reason(batch)
        report = engine.explain(result, sample_idx=0)

        assert isinstance(report, str)
        assert "1단계" in report
        assert "2단계" in report
        assert "3단계" in report
        assert "4단계" in report
        assert "최종 판정" in report
