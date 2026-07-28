"""Multi-task loss functions for joint training."""

import torch
import torch.nn as nn


class MultitaskGaitLoss(nn.Module):
    """Weighted multi-task loss for gait analysis.

    Combines:
      - Gait classification (CrossEntropy)
      - Disease classification (CrossEntropy) + Severity regression (MSE)
      - Fall risk classification (CrossEntropy) + Risk score regression (MSE)
      - Gait phase detection (CrossEntropy, per-frame)

    Supports uncertainty-based automatic weight balancing
    (Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses").
    """

    def __init__(
        self,
        active_tasks: list = None,
        task_weights: dict = None,
        use_uncertainty_weighting: bool = False,
    ):
        super().__init__()
        self.active_tasks = active_tasks or [
            "gait", "disease", "fall_risk", "gait_phase",
        ]

        # Fixed weights (used when uncertainty weighting is off)
        self.task_weights = task_weights or {
            "gait": 1.0,
            "disease": 1.0,
            "fall_risk": 1.5,  # higher weight for safety-critical task
            "severity": 0.5,
            "risk_score": 0.5,
            "gait_phase": 0.8,
        }

        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            # Learnable log-variance parameters (one per task)
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(1))
                for task in self.active_tasks
            })

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs: dict, targets: dict) -> dict:
        """
        Args:
            outputs: Model output dict from MultitaskGaitNet.
            targets: Dict with ground truth for each task.

        Returns:
            Dict with individual losses and total loss.
        """
        losses = {}

        # ── Gait classification ──
        if "gait" in self.active_tasks and "gait_logits" in outputs:
            losses["gait"] = self.ce_loss(
                outputs["gait_logits"], targets["gait_label"]
            )

        # ── Disease classification + severity ──
        if "disease" in self.active_tasks and "disease_logits" in outputs:
            losses["disease"] = self.ce_loss(
                outputs["disease_logits"], targets["disease_label"]
            )
            if "severity" in outputs and "severity_target" in targets:
                losses["severity"] = self.mse_loss(
                    outputs["severity"].squeeze(-1),
                    targets["severity_target"].float(),
                )

        # ── Fall risk ──
        if "fall_risk" in self.active_tasks and "risk_logits" in outputs:
            losses["fall_risk"] = self.ce_loss(
                outputs["risk_logits"], targets["fall_risk_label"]
            )
            if "risk_score" in outputs and "risk_score_target" in targets:
                losses["risk_score"] = self.mse_loss(
                    outputs["risk_score"].squeeze(-1),
                    targets["risk_score_target"].float(),
                )

        # ── Gait phase detection ──
        if "gait_phase" in self.active_tasks and "phase_logits" in outputs:
            # phase_logits: (B, num_phases, T), phase_targets: (B, T)
            losses["gait_phase"] = self.ce_loss(
                outputs["phase_logits"], targets["phase_label"]
            )

        # ── Combine losses ──
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)

        if self.use_uncertainty_weighting:
            for task, loss in losses.items():
                task_key = task if task in self.log_vars else task.split("_")[0]
                if task_key in self.log_vars:
                    precision = torch.exp(-self.log_vars[task_key])
                    total = total + precision * loss + self.log_vars[task_key]
                else:
                    total = total + loss
        else:
            for task, loss in losses.items():
                weight = self.task_weights.get(task, 1.0)
                total = total + weight * loss

        losses["total"] = total
        return losses
