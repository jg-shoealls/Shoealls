import torch
import numpy as np
import pandas as pd
import time
import json
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.data.ces_processor import CESDataProcessor
from src.models.gait_lstm import GaitLSTM, GaitWindowDataset
from torch.utils.data import DataLoader

def run_evaluation():
    print("🚀 CES 혁신상 제출용 정량적 평가 시작...")
    
    # 1. 데이터 및 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CESDataProcessor('CES.csv')
    df = processor.load_and_preprocess()
    
    features = processor.get_feature_matrix()
    labels = processor.get_labels()
    
    # 클래스 이름 추출
    class_names = list(processor.label_encoder.classes_)
    
    # 2. 데이터셋 준비 (테스트용)
    window_size = 10
    dataset = GaitWindowDataset(features, labels, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # 추론 시간 측정을 위해 batch=1
    
    # 3. 모델 초기화 및 가중치 로드
    input_dim = features.shape[1]
    model = GaitLSTM(input_dim, hidden_dim=64, num_layers=2, num_classes=len(class_names)).to(device)
    
    checkpoint_path = 'outputs/ces/gait_lstm_ces.pt'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ 모델 가중치 로드 완료 ({checkpoint_path})")
    except Exception as e:
        print(f"⚠️ 경고: 저장된 모델을 찾을 수 없습니다. ({e})")

    model.eval()
    
    all_preds = []
    all_labels = []
    latencies = []

    # 4. 정밀 평가 수행
    print(f"📊 {len(dataloader)}개 샘플 추론 중...")
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            y = batch["label"].to(device)
            
            # 지연 시간 측정 (Latency)
            start_time = time.perf_counter()
            logits, _, _ = model(x)
            end_time = time.perf_counter()
            
            latencies.append(end_time - start_time)
            
            pred = torch.argmax(logits, dim=1)
            all_preds.append(pred.item())
            all_labels.append(y.item())

    # 5. 지표 계산
    unique_labels = np.unique(all_labels)
    present_class_names = [class_names[i] for i in unique_labels]
    
    # 모든 클래스에 대해 지표를 계산하도록 labels 매개변수 명시
    report = classification_report(
        all_labels, 
        all_preds, 
        labels=range(len(class_names)), 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    
    avg_latency = np.mean(latencies) * 1000 # ms
    p99_latency = np.percentile(latencies, 99) * 1000 # ms
    
    # 6. 결과 출력 (혁신상 보고서용)
    print("\n" + "="*50)
    print("🏆 [CES Innovation Award - Technical Metrics]")
    print("="*50)
    print(f"1. 전체 정확도 (Accuracy): {report['accuracy']*100:.2f}%")
    print(f"2. 평균 F1-Score: {report['macro avg']['f1-score']:.4f}")
    print("\n[클래스별 상세 지표]")
    for cls in class_names:
        f1 = report[cls]['f1-score']
        print(f" - {cls:10}: F1={f1:.4f}, Precision={report[cls]['precision']:.4f}, Recall={report[cls]['recall']:.4f}")
    
    print("\n[엔지니어링 성능 - 실시간성]")
    print(f" - 평균 추론 지연 시간: {avg_latency:.2f} ms")
    print(f" - P99 지연 시간 (최악의 경우): {p99_latency:.2f} ms")
    print(f" - 시스템 초당 처리량 (Throughput): {1000/avg_latency:.1f} inferences/sec")
    print("="*50)

    # JSON 결과 저장
    results = {
        "metrics": report,
        "latency": {
            "avg_ms": avg_latency,
            "p99_ms": p99_latency
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open('outputs/ces/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("💾 결과가 'outputs/ces/evaluation_results.json'에 저장되었습니다.")

if __name__ == "__main__":
    import os
    os.makedirs('outputs/ces', exist_ok=True)
    run_evaluation()
