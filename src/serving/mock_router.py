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

@router.post("/analyze")
async def analyze(payload: dict):
    # Determine the profile from some basic heuristic or just return demo data
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
            "priority_actions": ["보행 보조기구 사용 권장", "보호자 동반 보행"]
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
            "report_kr": "파킨슨성 보행 패턴이 강하게 의심됩니다.",
            "is_demo_mode": True
        }
    }
