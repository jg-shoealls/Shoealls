"""WearGait-PD 데이터셋 전용 학습 스크립트.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

from src.data.weargait_adapter import WearGaitPDAdapter
from src.models.multimodal_gait_net import MultimodalGaitNet
from src.training.train import train_one_epoch, evaluate
from src.utils.metrics import compute_metrics
from src.validation.report import generate_report

def main():
    parser = argparse.ArgumentParser(description="WearGait-PD 데이터셋으로 멀티모달 보행 AI 학습")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="WearGait-PD 데이터 루트 (HC, PD 폴더 포함)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="설정 파일 경로")
    parser.add_argument("--output-dir", type=str, default="outputs/weargait", help="결과 저장 경로")
    parser.add_argument("--task", type=str, default="SelfPace", help="학습할 태스크 (SelfPace 또는 TUG)")
    args = parser.parse_args()

    # 1. 설정 로드
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # WearGait-PD 특성에 맞게 일부 설정 조정
    config["data"]["num_classes"] = 4 # normal, antalgic, ataxic, parkinsonian (WearGait는 0과 3만 사용)
    config["data"]["class_names"] = ["Normal", "Antalgic", "Ataxic", "Parkinsonian"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 데이터 로드
    print("=" * 60)
    print(f"STEP 1: WearGait-PD 데이터 로드 (Task: {args.task})")
    print("=" * 60)
    adapter = WearGaitPDAdapter(args.data_dir)
    dataset = adapter.to_dataset(sequence_length=config["data"]["sequence_length"])
    print(f"총 샘플 수: {len(dataset)}")

    # 3. 데이터 분할
    total = len(dataset)
    train_n = int(total * 0.7)
    val_n = int(total * 0.15)
    test_n = total - train_n - val_n
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"분할: 학습 {len(train_ds)} / 검증 {len(val_ds)} / 테스트 {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"])

    # 4. 모델 학습
    print("\n" + "=" * 60)
    print("STEP 2: 모델 학습 시작")
    print("=" * 60)
    model = MultimodalGaitNet(config).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, config["training"]["epochs"] + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(train_m["accuracy"])
        history["val_acc"].append(val_m["accuracy"])

        print(f"Epoch {epoch:3d} | Train Acc: {train_m['accuracy']:.4f} | Val Acc: {val_m['accuracy']:.4f} | {time.time()-t0:.1f}s")

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_accuracy": best_val_acc,
                "model_type": "basic"
            }, output_dir / "best_model.pt")

    # 5. 최종 평가 및 보고서
    print("\n" + "=" * 60)
    print("STEP 3: 최종 평가")
    print("=" * 60)
    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_m = evaluate(model, test_loader, criterion, device)
    print(f"최종 테스트 정확도: {test_m['accuracy']:.4f}")

    # model_manager를 사용하여 모델 등록
    from src.utils.model_manager import model_manager
    model_id = model_manager.save_model(
        model_state=checkpoint["model_state_dict"],
        config=config,
        metrics={"test_accuracy": float(test_m["accuracy"]), "val_accuracy": float(best_val_acc)},
        version="1.0.0",
        model_type="basic",
        alias="production"
    )
    print(f"모델이 레지스트리에 등록되었습니다: {model_id} (alias: production)")

    print(f"\n학습 완료! 결과가 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()
