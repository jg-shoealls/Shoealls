"""
CES.csv 데이터 기반 보행 분류 모델 학습 스크립트.

실행 예시:
    python scripts/train_ces.py --data CES.csv --epochs 30 --batch-size 64
    python scripts/train_ces.py --data CES.csv --epochs 50 --ensemble --save outputs/ces_ensemble.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 프로젝트 루트를 sys.path에 추가 (상대 임포트 보장)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ces_dataset import (
    FEATURE_COLS,
    DataLoaderConfig,
    DataProcessor,
    ProcessorConfig,
    build_dataloaders,
)
from src.models.ces_ensemble import ShoeallasCESEnsemble, train_ensemble_epoch
from src.models.gait_lstm import (
    CombinedGaitLoss,
    GaitLSTM,
    TrainStepResult,
    evaluate,
    train_one_epoch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── 인수 파서 ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shoealls CES 보행 분류 학습")
    p.add_argument("--data", default="CES.csv", help="CES.csv 경로")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--window-size", type=int, default=30, help="슬라이딩 윈도우 프레임 수")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-fall", type=float, default=2.0, help="낙상 감지 손실 가중치")
    p.add_argument("--lambda-ae", type=float, default=0.3, help="AE 재구성 손실 가중치 (앙상블 전용)")
    p.add_argument("--ensemble", action="store_true", help="앙상블 모드 (AE + LSTM) 학습")
    p.add_argument("--save", default="outputs/ces_model.pt", help="체크포인트 저장 경로")
    p.add_argument("--device", default="auto")
    return p.parse_args()


# ─── 데이터 준비 ─────────────────────────────────────────────────────────────


def prepare_data(args: argparse.Namespace):
    """CES.csv 로드 → 전처리 → DataLoader 생성"""
    logger.info("데이터 로드 중: %s", args.data)
    processor = DataProcessor(ProcessorConfig())
    processor.load(args.data)
    df = processor.process()

    summary = processor.summary(df)
    logger.info("데이터 통계: %s", json.dumps(summary, ensure_ascii=False, default=str))

    # 클래스 불균형 경고
    label_dist = summary.get("label_dist", {})
    if label_dist:
        total = sum(label_dist.values())
        for cls, cnt in label_dist.items():
            ratio = cnt / total * 100
            if ratio < 5:
                logger.warning("클래스 '%s' 비율이 매우 낮음: %.1f%% — 불균형 주의", cls, ratio)

    loader_cfg = DataLoaderConfig(
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        feature_cols=FEATURE_COLS,
    )
    train_loader, val_loader = build_dataloaders(df, loader_cfg)
    n_features = len([c for c in FEATURE_COLS if c in df.columns])
    return train_loader, val_loader, n_features


# ─── LSTM 전용 학습 ──────────────────────────────────────────────────────────


def train_lstm(args: argparse.Namespace, train_loader, val_loader, n_features: int, device: torch.device):
    logger.info("=== GaitLSTM 학습 시작 ===")
    model = GaitLSTM(
        input_dim=n_features,
        hidden_dim=args.hidden_dim,
        num_classes=3,
        bidirectional=True,
    ).to(device)

    criterion = CombinedGaitLoss(lambda_fall=args.lambda_fall)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        result: TrainStepResult = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        log_msg = (
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={result.loss:.4f} | "
            f"fall_bce={result.fall_bce:.4f} | "
            f"train_acc={result.accuracy:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"fall_recall={val_metrics['fall_recall']:.4f} | "
            f"lr={lr_now:.2e}"
        )
        logger.info(log_msg)
        history.append({
            "epoch": epoch,
            "loss": result.loss,
            "train_acc": result.accuracy,
            "val_acc": val_metrics["accuracy"],
            "fall_recall": val_metrics["fall_recall"],
        })

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    logger.info("최고 검증 정확도: %.4f", best_acc)
    return model, best_state, history


# ─── 앙상블 학습 ─────────────────────────────────────────────────────────────


def train_ensemble_model(args: argparse.Namespace, train_loader, val_loader, n_features: int, device: torch.device):
    logger.info("=== ShoeallasCESEnsemble 학습 시작 ===")
    model = ShoeallasCESEnsemble(n_features=n_features, lstm_hidden=args.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        metrics = train_ensemble_epoch(model, train_loader, optimizer, device, lambda_ae=args.lambda_ae)
        val_metrics = evaluate(model.classifier, val_loader, device)
        scheduler.step()

        logger.info(
            "Epoch %03d/%d | ce=%.4f | ae=%.4f | train_acc=%.4f | val_acc=%.4f | fall_recall=%.4f",
            epoch, args.epochs,
            metrics["ce_loss"], metrics["ae_loss"],
            metrics["accuracy"], val_metrics["accuracy"], val_metrics["fall_recall"],
        )
        history.append({"epoch": epoch, **metrics, **val_metrics})

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    logger.info("최고 검증 정확도: %.4f", best_acc)
    return model, best_state, history


# ─── 저장 ────────────────────────────────────────────────────────────────────


def save_checkpoint(
    model: torch.nn.Module,
    best_state: dict,
    history: list,
    args: argparse.Namespace,
    n_features: int,
) -> None:
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "model_type": "ensemble" if args.ensemble else "lstm",
            "n_features": n_features,
            "hidden_dim": args.hidden_dim,
            "window_size": args.window_size,
            "n_classes": 3,
            "feature_cols": FEATURE_COLS,
            "history": history,
            "args": vars(args),
        },
        save_path,
    )
    logger.info("체크포인트 저장: %s", save_path)

    # 이력 JSON 별도 저장
    history_path = save_path.with_suffix(".history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    logger.info("학습 이력 저장: %s", history_path)


# ─── 메인 ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # 디바이스 설정
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("학습 디바이스: %s", device)

    # 재현성
    torch.manual_seed(42)
    np.random.seed(42)

    # 데이터 준비
    train_loader, val_loader, n_features = prepare_data(args)
    logger.info("특성 수: %d, window_size=%d", n_features, args.window_size)

    # 학습
    if args.ensemble:
        model, best_state, history = train_ensemble_model(
            args, train_loader, val_loader, n_features, device
        )
    else:
        model, best_state, history = train_lstm(
            args, train_loader, val_loader, n_features, device
        )

    # 저장
    save_checkpoint(model, best_state, history, args, n_features)
    logger.info("학습 완료!")


if __name__ == "__main__":
    main()
