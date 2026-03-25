"""Tests for multi-task gait analysis model."""

import torch
import yaml

from src.models.multitask_gait_net import MultitaskGaitNet
from src.models.task_heads import (
    DiseaseClassificationHead,
    FallRiskPredictionHead,
    GaitPhaseDetectionHead,
)
from src.training.multitask_loss import MultitaskGaitLoss


def load_config():
    with open("configs/multitask.yaml") as f:
        return yaml.safe_load(f)


def make_batch(batch_size=2):
    return {
        "imu": torch.randn(batch_size, 6, 128),
        "pressure": torch.randn(batch_size, 128, 1, 16, 8),
        "skeleton": torch.randn(batch_size, 3, 128, 17),
    }


class TestTaskHeads:
    def test_disease_head(self):
        head = DiseaseClassificationHead(embed_dim=128, num_diseases=7)
        fused = torch.randn(2, 128)
        out = head(fused)
        assert out["disease_logits"].shape == (2, 7)
        assert out["severity"].shape == (2, 1)
        assert (out["severity"] >= 0).all() and (out["severity"] <= 1).all()

    def test_fall_risk_head(self):
        head = FallRiskPredictionHead(embed_dim=128)
        fused = torch.randn(2, 128)
        temporal = torch.randn(2, 50, 128)
        out = head(fused, temporal)
        assert out["risk_logits"].shape == (2, 2)
        assert out["risk_score"].shape == (2, 1)
        assert out["time_to_fall"].shape == (2, 1)
        assert (out["risk_score"] >= 0).all() and (out["risk_score"] <= 1).all()
        assert (out["time_to_fall"] >= 0).all()

    def test_gait_phase_head(self):
        head = GaitPhaseDetectionHead(embed_dim=128, num_phases=8)
        temporal = torch.randn(2, 100, 128)
        out = head(temporal)
        assert out["phase_logits"].shape == (2, 8, 100)


class TestMultitaskGaitNet:
    def test_forward_all_tasks(self):
        config = load_config()
        model = MultitaskGaitNet(config)
        batch = make_batch()
        outputs = model(batch)

        assert "gait_logits" in outputs
        assert "disease_logits" in outputs
        assert "severity" in outputs
        assert "risk_logits" in outputs
        assert "risk_score" in outputs
        assert "time_to_fall" in outputs
        assert "phase_logits" in outputs

        assert outputs["gait_logits"].shape == (2, 4)
        assert outputs["disease_logits"].shape == (2, 7)
        assert outputs["risk_logits"].shape == (2, 2)

    def test_selective_tasks(self):
        config = load_config()
        config["tasks"]["active"] = ["gait", "disease"]
        model = MultitaskGaitNet(config)
        batch = make_batch()
        outputs = model(batch)

        assert "gait_logits" in outputs
        assert "disease_logits" in outputs
        assert "risk_logits" not in outputs
        assert "phase_logits" not in outputs

    def test_param_breakdown(self):
        config = load_config()
        model = MultitaskGaitNet(config)
        breakdown = model.get_task_param_breakdown()

        assert "shared_encoders" in breakdown
        assert "shared_fusion" in breakdown
        assert "task_heads" in breakdown
        assert breakdown["total"] > 0

        # Task heads should be small relative to shared backbone
        shared = (
            sum(breakdown["shared_encoders"].values())
            + breakdown["shared_fusion"]
        )
        heads = sum(breakdown["task_heads"].values())
        assert shared > heads, "Shared backbone should have more params than heads"


class TestMultitaskLoss:
    def test_fixed_weight_loss(self):
        config = load_config()
        model = MultitaskGaitNet(config)
        loss_fn = MultitaskGaitLoss(
            active_tasks=["gait", "disease", "fall_risk"],
            use_uncertainty_weighting=False,
        )

        batch = make_batch(4)
        outputs = model(batch)
        targets = {
            "gait_label": torch.randint(0, 4, (4,)),
            "disease_label": torch.randint(0, 7, (4,)),
            "severity_target": torch.rand(4),
            "fall_risk_label": torch.randint(0, 2, (4,)),
            "risk_score_target": torch.rand(4),
        }

        losses = loss_fn(outputs, targets)
        assert "total" in losses
        assert "gait" in losses
        assert "disease" in losses
        assert "fall_risk" in losses
        assert losses["total"].requires_grad

    def test_uncertainty_weight_loss(self):
        loss_fn = MultitaskGaitLoss(
            active_tasks=["gait", "disease"],
            use_uncertainty_weighting=True,
        )

        outputs = {
            "gait_logits": torch.randn(4, 4),
            "disease_logits": torch.randn(4, 7),
        }
        targets = {
            "gait_label": torch.randint(0, 4, (4,)),
            "disease_label": torch.randint(0, 7, (4,)),
        }

        losses = loss_fn(outputs, targets)
        assert losses["total"].requires_grad

        # log_vars should be learnable
        assert len(list(loss_fn.log_vars.parameters())) > 0

    def test_backward_pass(self):
        config = load_config()
        model = MultitaskGaitNet(config)
        loss_fn = MultitaskGaitLoss(use_uncertainty_weighting=True)

        batch = make_batch(4)
        outputs = model(batch)

        targets = {
            "gait_label": torch.randint(0, 4, (4,)),
            "disease_label": torch.randint(0, 7, (4,)),
            "severity_target": torch.rand(4),
            "fall_risk_label": torch.randint(0, 2, (4,)),
            "risk_score_target": torch.rand(4),
            "phase_label": torch.randint(0, 8, (4, outputs["phase_logits"].shape[2])),
        }

        losses = loss_fn(outputs, targets)
        losses["total"].backward()

        # Key parameters should have gradients (some heads like time_to_fall
        # may not be in the loss, so we check critical components only)
        critical_modules = [
            "imu_encoder", "pressure_encoder", "skeleton_encoder",
            "fusion", "gait_classifier", "disease_head.classifier",
            "fall_risk_head.risk_classifier",
        ]
        for name, param in model.named_parameters():
            if param.requires_grad and any(m in name for m in critical_modules):
                assert param.grad is not None, f"No gradient for {name}"
