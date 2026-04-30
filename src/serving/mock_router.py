from fastapi import APIRouter
import numpy as np

router = APIRouter(prefix="/api/v1")

@router.get("/sample")
async def get_sample(gait_profile: str = "normal"):
    # Return dummy sensor data for the dashboard demo
    imu = np.zeros((128, 6)).tolist()
    pressure = np.zeros((16, 8)).tolist()
    skeleton = np.zeros((128, 17, 3)).tolist()
    
    return {
        "gait_profile": gait_profile,
        "sensor_data": {
            "imu": imu,
            "pressure": pressure,
            "skeleton": skeleton
        },
        "features": {
            "cadence": 105.0,
            "stride_length": 1.2
        }
    }

import asyncio
import ollama

@router.post("/analyze")
async def analyze(payload: dict):
    # Determine the profile from some basic heuristic or just return demo data
    
    # Context for LLM prompt
    disease = "파킨슨병 (Parkinson's)"
    abnormal_biomarkers = "종종걸음, 안정시 떨림, 비대칭적인 입각기"
    fall_risk = "고위험 (85%)"
    
    # Prompt 1: Generate reasoning report
    prompt_report = f"""
당신은 세계 최고의 보행 분석 AI 전문가입니다.
다음 보행 분석 결과를 바탕으로 환자에게 제공할 '종합 소견서'를 3문장 이내의 자연스러운 한국어로 작성하세요.

분석 결과:
- 주요 의심 질환: {disease}
- 비정상 생체 지표: {abnormal_biomarkers}
- 낙상 위험도: {fall_risk}

출력은 오직 소견서 텍스트만 포함하세요.
"""
    # Prompt 2: Generate priority actions
    prompt_actions = f"""
당신은 재활 치료사입니다.
다음 보행 분석 결과를 바탕으로 환자가 즉각적으로 취해야 할 '우선 권장 행동' 3가지를 한국어로 작성하세요.

분석 결과:
- 주요 의심 질환: {disease}
- 낙상 위험도: {fall_risk}

출력은 '-'로 시작하는 목록 형태로 3줄만 작성하세요. 다른 말은 절대 추가하지 마세요.
"""
    
    # Create the client
    client = ollama.AsyncClient()
    
    # Function to call LLM safely
    async def get_llm_response(prompt: str, fallback: str):
        try:
            response = await client.chat(model='llama3', messages=[
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Ollama error: {e}")
            return fallback

    # Run both prompts concurrently
    report_kr, actions_text = await asyncio.gather(
        get_llm_response(prompt_report, "파킨슨성 보행 패턴이 강하게 의심됩니다. 신경과 진료를 권장합니다."),
        get_llm_response(prompt_actions, "- 보행 보조기구 사용 권장\n- 보호자 동반 보행\n- 즉각적인 신경과 검진")
    )
    
    # Parse the actions text into a list
    priority_actions = [line.strip("- ").strip() for line in actions_text.split("\n") if line.strip()]
    if not priority_actions:
        priority_actions = ["보행 보조기구 사용 권장", "보호자 동반 보행"]

    return {
        "classify": {
            "prediction": "parkinsonian",
            "prediction_kr": "파킨슨",
            "confidence": 0.92,
            "class_probabilities": {"normal": 0.05, "antalgic": 0.02, "ataxic": 0.01, "parkinsonian": 0.92},
            "is_demo_mode": True
        },
        "disease_risk": {
            "top_diseases": [
                {"disease": "parkinsons", "disease_kr": "파킨슨병", "risk_score": 0.88, "severity": "고위험", "key_signs": ["종종걸음", "안정시 떨림"], "referral": "신경과"}
            ],
            "ml_prediction": "parkinsons",
            "ml_prediction_kr": "파킨슨병",
            "ml_confidence": 0.88,
            "ml_top3": [
                {"name_kr": "파킨슨병", "probability": 0.88},
                {"name_kr": "소뇌 실조증", "probability": 0.12},
                {"name_kr": "정상", "probability": 0.0}
            ],
            "abnormal_biomarkers": ["종종걸음", "보폭 감소"]
        },
        "injury_risk": {
            "predicted_injury": "fall_risk",
            "predicted_injury_kr": "낙상 위험",
            "confidence": 0.85,
            "top3": [
                {"name_kr": "낙상", "probability": 0.85},
                {"name_kr": "발목 염좌", "probability": 0.1},
                {"name_kr": "무릎 통증", "probability": 0.05}
            ],
            "combined_risk_score": 0.85,
            "combined_risk_grade": "고위험",
            "timeline": "즉각적인 주의 필요",
            "priority_actions": priority_actions
        },
        "reasoning": {
            "final_prediction": "parkinsonian",
            "final_prediction_kr": "파킨슨성 보행",
            "confidence": 0.92,
            "reasoning_trace": [
                {"step": 1, "label": "IMU 분석", "prediction": "abnormal", "prediction_kr": "비정상", "probability": 0.95},
                {"step": 2, "label": "족저압 분석", "prediction": "asymmetric", "prediction_kr": "비대칭", "probability": 0.88},
                {"step": 3, "label": "종합 추론", "prediction": "parkinsonian", "prediction_kr": "파킨슨", "probability": 0.92}
            ],
            "anomaly_findings": [
                {"modality": "imu", "anomalies": [{"type": "tremor", "score": 0.8}]}
            ],
            "uncertainty": 0.08,
            "evidence_strength": 0.92,
            "report_kr": report_kr,
            "is_demo_mode": True
        }
    }
