"""DL 모델 파인튜닝 파이프라인.

사전학습된 MultimodalGaitNet을 실제/합성 데이터로 파인튜닝합니다.

3단계 점진적 언프리징 전략:
  Phase 1: 인코더 동결 → 분류 헤드만 학습 (빠른 수렴)
  Phase 2: 퓨전 + 분류기 학습 (중간 계층 적응)
  Phase 3: 전체 모델 미세 조정 (판별적 학습률)

사용법:
    finetuner = GaitModelFineTuner(config, pretrained_path="best_model.pt")
    result = finetuner.run(train_loader, val_loader)
    finetuner.save(result, "outputs/finetuned")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.multimodal_gait_net import MultimodalGaitNet
from src.utils.metrics import compute_metrics


@dataclass
class FinetunePhase:
    """파인튜닝 단계 설정."""
    name: str
    epochs: int
    lr: float
    unfreeze: list[str]
    weight_decay: float = 1e-4
    warmup_epochs: int = 2


@dataclass
class FinetuneResult:
    """파인튜닝 결과."""
    phases_completed: int
    total_epochs: int
    best_val_accuracy: float
    best_val_f1: float
    best_epoch: int
    history: dict[str, list[float]] = field(default_factory=dict)
    final_test_metrics: Optional[dict] = None


class GaitModelFineTuner:
    """멀티모달 보행 분석 모델 파인튜너.

    사전학습 모델 로드 → 3단계 점진적 언프리징 → 최적 체크포인트 저장.
    """

    DEFAULT_PHASES = [
        FinetunePhase(
            name="head_only",
            epochs=10,
            lr=1e-3,
            unfreeze=["classifier"],
        ),
        FinetunePhase(
            name="fusion_and_head",
            epochs=15,
            lr=3e-4,
            unfreeze=["fusion", "classifier"],
        ),
        FinetunePhase(
            name="full_model",
            epochs=20,
            lr=1e-4,
            unfreeze=["imu_encoder", "pressure_encoder", "skeleton_encoder", "fusion", "classifier"],
            weight_decay=1e-3,
        ),
    ]

    ENCODER_NAMES = ["imu_encoder", "pressure_encoder", "skeleton_encoder"]
    ALL_MODULES = ENCODER_NAMES + ["fusion", "classifier"]

    def __init__(
        self,
        config: dict,
        pretrained_path: Optional[str] = None,
        phases: Optional[list[FinetunePhase]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.phases = phases or self.DEFAULT_PHASES
        self.model = MultimodalGaitNet(config).to(self.device)

        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str):
        """사전학습 체크포인트 로드 (분류 헤드 크기 불일치 허용)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        model_state = self.model.state_dict()
        loaded, skipped = [], []
        for key, val in state_dict.items():
            if key in model_state and model_state[key].shape == val.shape:
                model_state[key] = val
                loaded.append(key)
            else:
                skipped.append(key)

        self.model.load_state_dict(model_state)
        print(f"Pretrained: loaded {len(loaded)} params, skipped {len(skipped)}")
        if skipped:
            print(f"  Skipped: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    def _freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def _unfreeze_modules(self, module_names: list[str]):
        for name in module_names:
            module = getattr(self.model, name, None)
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = True

    def _build_optimizer(self, phase: FinetunePhase) -> torch.optim.Optimizer:
        """판별적 학습률로 옵티마이저 생성.

        언프리즈된 인코더에는 더 낮은 학습률, 헤드에는 더 높은 학습률 적용.
        """
        param_groups = []

        for name in phase.unfreeze:
            module = getattr(self.model, name, None)
            if module is None:
                continue
            params = [p for p in module.parameters() if p.requires_grad]
            if not params:
                continue

            if name in self.ENCODER_NAMES:
                lr = phase.lr * 0.1
            elif name == "fusion":
                lr = phase.lr * 0.5
            else:
                lr = phase.lr

            param_groups.append({"params": params, "lr": lr, "name": name})

        if not param_groups:
            param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": phase.lr}]

        return torch.optim.AdamW(
            param_groups,
            weight_decay=phase.weight_decay,
        )

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("label")

            logits = self.model(batch)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / len(all_preds)
        return metrics

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("label")

            logits = self.model(batch)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / len(all_preds)
        return metrics

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> FinetuneResult:
        """3단계 파인튜닝 실행."""
        criterion = nn.CrossEntropyLoss()
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": [],
            "phase": [], "lr": [],
        }

        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = 0
        best_state = None
        total_epoch = 0

        for phase_idx, phase in enumerate(self.phases):
            print(f"\n{'='*60}")
            print(f"  Phase {phase_idx+1}/{len(self.phases)}: {phase.name}")
            print(f"  LR: {phase.lr}, Epochs: {phase.epochs}")
            print(f"  Unfreeze: {phase.unfreeze}")
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())

            self._freeze_all()
            self._unfreeze_modules(phase.unfreeze)

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
            print(f"{'='*60}")

            optimizer = self._build_optimizer(phase)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(phase.epochs - phase.warmup_epochs, 1),
            )

            patience = 0
            phase_best = 0.0

            for epoch in range(1, phase.epochs + 1):
                total_epoch += 1
                t0 = time.time()

                train_m = self._train_epoch(train_loader, criterion, optimizer)
                val_m = self._evaluate(val_loader, criterion)

                if epoch > phase.warmup_epochs:
                    scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                history["train_loss"].append(train_m["loss"])
                history["val_loss"].append(val_m["loss"])
                history["train_acc"].append(train_m["accuracy"])
                history["val_acc"].append(val_m["accuracy"])
                history["train_f1"].append(train_m["f1_macro"])
                history["val_f1"].append(val_m["f1_macro"])
                history["phase"].append(phase.name)
                history["lr"].append(current_lr)

                elapsed = time.time() - t0
                print(
                    f"  [{phase.name}] Epoch {epoch}/{phase.epochs} | "
                    f"Train F1: {train_m['f1_macro']:.4f} | "
                    f"Val F1: {val_m['f1_macro']:.4f} Acc: {val_m['accuracy']:.4f} | "
                    f"LR: {current_lr:.2e} | {elapsed:.1f}s"
                )

                if val_m["accuracy"] > best_val_acc:
                    best_val_acc = val_m["accuracy"]
                    best_val_f1 = val_m["f1_macro"]
                    best_epoch = total_epoch
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

                if val_m["f1_macro"] > phase_best + 0.001:
                    phase_best = val_m["f1_macro"]
                    patience = 0
                else:
                    patience += 1

                if patience >= 5:
                    print(f"  Early stopping at phase epoch {epoch}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        result = FinetuneResult(
            phases_completed=len(self.phases),
            total_epochs=total_epoch,
            best_val_accuracy=best_val_acc,
            best_val_f1=best_val_f1,
            best_epoch=best_epoch,
            history=history,
        )

        if test_loader:
            test_m = self._evaluate(test_loader, criterion)
            result.final_test_metrics = test_m
            print(f"\n{'='*60}")
            print(f"  Final Test — Acc: {test_m['accuracy']:.4f} F1: {test_m['f1_macro']:.4f}")
            print(f"{'='*60}")

        return result

    def save(self, result: FinetuneResult, output_dir: str):
        """체크포인트 + 학습 이력 저장."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "best_val_accuracy": result.best_val_accuracy,
            "best_val_f1": result.best_val_f1,
            "best_epoch": result.best_epoch,
            "history": result.history,
            "test_metrics": result.final_test_metrics,
        }, out / "finetuned_model.pt")

        try:
            self._plot_history(result.history, out)
        except Exception:
            pass

        print(f"Saved to {out}")

    def _plot_history(self, history: dict, output_dir: Path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)
        phases = history["phase"]
        phase_boundaries = []
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                phase_boundaries.append(i)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(epochs, history["train_loss"], label="Train", alpha=0.8)
        axes[0, 0].plot(epochs, history["val_loss"], label="Val", alpha=0.8)
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, history["train_acc"], label="Train", alpha=0.8)
        axes[0, 1].plot(epochs, history["val_acc"], label="Val", alpha=0.8)
        axes[0, 1].set_title("Accuracy")
        axes[0, 1].legend()

        axes[1, 0].plot(epochs, history["train_f1"], label="Train", alpha=0.8)
        axes[1, 0].plot(epochs, history["val_f1"], label="Val", alpha=0.8)
        axes[1, 0].set_title("F1 Macro")
        axes[1, 0].legend()

        axes[1, 1].plot(epochs, history["lr"], color="green")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_yscale("log")

        for ax in axes.flat:
            for b in phase_boundaries:
                ax.axvline(b + 0.5, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.15)

        plt.suptitle("Fine-tuning History", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "finetune_history.png", dpi=150, bbox_inches="tight")
        plt.close()


def finetune_from_config(
    config_path: str = "configs/default.yaml",
    pretrained_path: Optional[str] = None,
    output_dir: str = "outputs/finetuned",
) -> FinetuneResult:
    """설정 파일 기반 파인튜닝 실행 (편의 함수)."""
    import yaml
    from src.training.train import create_dataloaders

    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader = create_dataloaders(config)

    finetuner = GaitModelFineTuner(
        config=config,
        pretrained_path=pretrained_path,
    )

    result = finetuner.run(train_loader, val_loader, test_loader)
    finetuner.save(result, output_dir)
    return result
