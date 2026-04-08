"""자동 하이퍼파라미터 튜닝 모듈.

Optuna를 활용하여 모델의 최적 하이퍼파라미터를 자동으로 탐색합니다.

Automatic hyperparameter tuning module using Optuna.
Searches for optimal hyperparameters with pruning support.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

try:
    import optuna
    from optuna.trial import Trial
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def create_objective(
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = 20,
):
    """Optuna objective 함수 생성.

    Args:
        config: 기본 설정 (탐색 범위 외 고정값).
        train_loader: 학습 데이터 로더.
        val_loader: 검증 데이터 로더.
        device: 디바이스.
        max_epochs: 최대 학습 에포크.

    Returns:
        Optuna objective 함수.
    """
    from src.models.multimodal_gait_net import MultimodalGaitNet

    def objective(trial: "Trial") -> float:
        # 하이퍼파라미터 탐색 공간 정의
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # embed_dim이 num_heads로 나누어져야 함
        if embed_dim % num_heads != 0:
            raise optuna.TrialPruned()

        # 설정 업데이트
        trial_config = _update_config(
            config, lr=lr, weight_decay=weight_decay, dropout=dropout,
            embed_dim=embed_dim, num_heads=num_heads,
        )

        # 모델 생성
        model = MultimodalGaitNet(trial_config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # 학습
        best_val_acc = 0.0
        for epoch in range(max_epochs):
            # Train
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("label")
                logits = model(batch)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    labels = batch.pop("label")
                    logits = model(batch)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total if total > 0 else 0.0
            best_val_acc = max(best_val_acc, val_acc)

            # Pruning: 성능이 좋지 않은 trial 조기 종료
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_acc

    return objective


def _update_config(
    config: dict,
    lr: float,
    weight_decay: float,
    dropout: float,
    embed_dim: int,
    num_heads: int,
) -> dict:
    """설정 딕셔너리를 trial 파라미터로 업데이트."""
    import copy
    c = copy.deepcopy(config)
    c["training"]["learning_rate"] = lr
    c["training"]["weight_decay"] = weight_decay
    c["model"]["fusion"]["embed_dim"] = embed_dim
    c["model"]["fusion"]["num_heads"] = num_heads
    c["model"]["imu_encoder"]["dropout"] = dropout
    c["model"]["pressure_encoder"]["dropout"] = dropout
    c["model"]["skeleton_encoder"]["dropout"] = dropout
    return c


def run_tuning(
    config: dict,
    n_trials: int = 50,
    max_epochs: int = 20,
    output_dir: str = "outputs/tuning",
) -> dict:
    """하이퍼파라미터 튜닝 실행.

    Args:
        config: 기본 설정.
        n_trials: 총 시도 횟수.
        max_epochs: trial당 최대 에포크.
        output_dir: 결과 저장 경로.

    Returns:
        best_params: 최적 하이퍼파라미터.
    """
    if not HAS_OPTUNA:
        raise ImportError("optuna가 설치되지 않았습니다: pip install optuna")

    from src.data.dataset import MultimodalGaitDataset
    from src.data.synthetic import generate_synthetic_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 데이터 준비
    data_cfg = config["data"]
    dataset_dict = generate_synthetic_dataset(
        num_samples_per_class=50,
        num_classes=data_cfg["num_classes"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )
    dataset = MultimodalGaitDataset(
        dataset_dict,
        sequence_length=data_cfg["sequence_length"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )

    train_n = int(len(dataset) * 0.8)
    val_n = len(dataset) - train_n
    train_ds, val_ds = random_split(
        dataset, [train_n, val_n],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # Optuna study 생성
    study = optuna.create_study(
        direction="maximize",
        study_name="gait_hyperparameter_tuning",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    objective = create_objective(config, train_loader, val_loader, device, max_epochs)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 결과 저장
    best_params = study.best_params
    best_value = study.best_value
    print(f"\nBest validation accuracy: {best_value:.4f}")
    print(f"Best parameters: {best_params}")

    # 최적 설정을 YAML로 저장
    best_config = _update_config(
        config,
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        dropout=best_params["dropout"],
        embed_dim=best_params["embed_dim"],
        num_heads=best_params["num_heads"],
    )
    with open(output_path / "best_config.yaml", "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)

    # 시각화
    try:
        _plot_optimization_history(study, output_path)
    except Exception:
        pass

    return {"best_params": best_params, "best_value": best_value, "study": study}


def _plot_optimization_history(study, output_path: Path):
    """최적화 이력 시각화."""
    import matplotlib.pyplot as plt

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return

    values = [t.value for t in trials]
    best_values = np.maximum.accumulate(values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Optimization history
    ax = axes[0]
    ax.scatter(range(len(values)), values, alpha=0.5, s=20, label="Trial")
    ax.plot(best_values, color="red", linewidth=2, label="Best")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Optimization History")
    ax.legend()

    # Parameter importance (simple correlation)
    ax2 = axes[1]
    param_names = list(trials[0].params.keys())
    importances = []
    for name in param_names:
        param_values = [t.params.get(name, 0) for t in trials]
        if isinstance(param_values[0], (int, float)):
            corr = abs(np.corrcoef(param_values, values)[0, 1])
            importances.append(corr if not np.isnan(corr) else 0)
        else:
            importances.append(0)

    sorted_idx = np.argsort(importances)[::-1]
    ax2.barh(
        [param_names[i] for i in sorted_idx],
        [importances[i] for i in sorted_idx],
        color="#2196F3",
    )
    ax2.set_xlabel("Correlation with Accuracy")
    ax2.set_title("Parameter Importance")

    plt.tight_layout()
    plt.savefig(output_path / "optimization_history.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="outputs/tuning")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_tuning(config, args.n_trials, args.max_epochs, args.output_dir)


if __name__ == "__main__":
    main()
