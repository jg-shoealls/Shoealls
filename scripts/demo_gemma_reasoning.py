import torch
import yaml
import json
import requests
from src.models.reasoning_engine import GaitReasoningEngine

def make_batch(batch_size=1):
    return {
        "imu": torch.randn(batch_size, 6, 128),
        "pressure": torch.randn(batch_size, 128, 1, 16, 8),
        "skeleton": torch.randn(batch_size, 3, 128, 17),
    }

def run_gemma_ollama_demo():
    print("="*60)
    print("  GEMMA-2-2B (via Ollama) x SHOEALLS GAIT ANALYSIS")
    print("="*60)

    # 1. Load Configuration and Engine
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = GaitReasoningEngine(config).to(device)
    engine.eval()

    # 2. Analyze with Shoealls Engine
    print("\n[1/2] Analyzing Gait Data with Custom Reasoning Engine...")
    batch = make_batch(batch_size=1)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.no_grad():
        analysis_result = engine.reason(batch)
    
    base_report = engine.explain(analysis_result, sample_idx=0)
    print("Base analysis complete.")

    # 3. Generate Report via Ollama
    print("\n[2/2] Generating Professional Report via Ollama (Gemma2:2b)...")
    
    prompt = f"""
당신은 보행 분석 전문가이자 전문의입니다. 
아래의 보행 분석 AI가 도출한 원시 리포트(Base Report)를 바탕으로, 환자나 보호자가 이해하기 쉬우면서도 의학적으로 전문적인 '종합 보행 분석 의견서'를 작성하세요.

[보행 분석 AI 결과 요약]
{base_report}

[지침]
1. 먼저 환자의 현재 상태를 친절하게 설명하세요.
2. 발견된 주요 이상 패턴(비대칭, 떨림 등)이 어떤 의미인지 의학적으로 해석하세요.
3. 최종 판정된 질환 가능성에 대해 설명하고, 주의해야 할 점을 제시하세요.
4. 마지막으로 권장되는 재활 운동이나 추가 검사에 대한 조언을 포함하세요.
5. 반드시 한국어로 작성하세요.

의견서:
"""

    # localhost 대신 127.0.0.1을 사용하여 연결성 강화
    url = "http://127.0.0.1:11434/api/generate"
    data = {
        "model": "gemma2:2b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        professional_report = result.get("response", "응답을 생성하지 못했습니다.")
        
        print("\n" + "="*60)
        print("  GEMMA(OLLAMA) GENERATED PROFESSIONAL REPORT")
        print("="*60)
        print(professional_report)
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\nError: Ollama 서비스에 연결할 수 없습니다. 'ollama serve'가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    run_gemma_ollama_demo()
