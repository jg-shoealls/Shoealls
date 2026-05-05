import asyncio
import logging
import os

import numpy as np
from fastapi import APIRouter
import ollama

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

_ollama_client = ollama.AsyncClient()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# 프로파일별 LLM 컨텍스트
_PROFILE_CONTEXT = {
    "normal": {
        "disease": "이상 없음 (정상 보행)",
        "abnormal_biomarkers": "없음",
        "fall_risk": "저위험 (5%)",
    },
    "parkinsons": {
        "disease": "파킨슨병 (Parkinson's Disease)",
        "abnormal_biomarkers": "종종걸음, 안정시 떨림, 비대칭적인 입각기",
        "fall_risk": "고위험 (85%)",
    },
    "stroke": {
        "disease": "뇌졸중 후유증 (Post-stroke)",
        "abnormal_biomarkers": "편측 보행 이상, 발 끌림, 비대칭 보폭",
        "fall_risk": "고위험 (78%)",
    },
    "fall_risk": {
        "disease": "낙상 고위험 (Fall Risk)",
        "abnormal_biomarkers": "균형 불안정, 보행 속도 저하, 지지 면적 증가",
        "fall_risk": "매우 고위험 (92%)",
    },
}

# 프로파일별 분류·진단·추론 목업 데이터
_PROFILE_RESPONSES = {
    "normal": {
        "classify": {
            "prediction": "normal",
            "prediction_kr": "정상 보행",
            "confidence": 0.95,
            "class_probabilities": {"normal": 0.95, "antalgic": 0.02, "ataxic": 0.02, "parkinsonian": 0.01},
            "is_demo_mode": True,
        },
        "disease_risk": {
            "top_diseases": [{"disease": "normal", "disease_kr": "정상", "risk_score": 0.05, "severity": "저위험", "key_signs": [], "referral": "해당없음"}],
            "ml_prediction": "normal", "ml_prediction_kr": "정상", "ml_confidence": 0.95,
            "ml_top3": [
                {"name_kr": "정상", "probability": 0.95},
                {"name_kr": "파킨슨병", "probability": 0.03},
                {"name_kr": "소뇌 실조증", "probability": 0.02},
            ],
            "abnormal_biomarkers": [],
        },
        "injury_risk": {
            "predicted_injury": "none", "predicted_injury_kr": "위험 없음", "confidence": 0.95,
            "top3": [
                {"name_kr": "위험 없음", "probability": 0.95},
                {"name_kr": "발목 염좌", "probability": 0.03},
                {"name_kr": "무릎 통증", "probability": 0.02},
            ],
            "combined_risk_score": 0.05, "combined_risk_grade": "저위험", "timeline": "현재 위험 없음",
        },
        "reasoning": {
            "final_prediction": "normal", "final_prediction_kr": "정상 보행", "confidence": 0.95,
            "reasoning_trace": [
                {"step": 1, "label": "IMU 분석", "prediction": "normal", "prediction_kr": "정상", "probability": 0.96},
                {"step": 2, "label": "족저압 분석", "prediction": "symmetric", "prediction_kr": "대칭", "probability": 0.94},
                {"step": 3, "label": "종합 추론", "prediction": "normal", "prediction_kr": "정상", "probability": 0.95},
            ],
            "anomaly_findings": [],
            "uncertainty": 0.05, "evidence_strength": 0.95, "is_demo_mode": True,
        },
    },
    "parkinsons": {
        "classify": {
            "prediction": "parkinsonian", "prediction_kr": "파킨슨", "confidence": 0.92,
            "class_probabilities": {"normal": 0.05, "antalgic": 0.02, "ataxic": 0.01, "parkinsonian": 0.92},
            "is_demo_mode": True,
        },
        "disease_risk": {
            "top_diseases": [{"disease": "parkinsons", "disease_kr": "파킨슨병", "risk_score": 0.88, "severity": "고위험", "key_signs": ["종종걸음", "안정시 떨림"], "referral": "신경과"}],
            "ml_prediction": "parkinsons", "ml_prediction_kr": "파킨슨병", "ml_confidence": 0.88,
            "ml_top3": [
                {"name_kr": "파킨슨병", "probability": 0.88},
                {"name_kr": "소뇌 실조증", "probability": 0.12},
                {"name_kr": "정상", "probability": 0.0},
            ],
            "abnormal_biomarkers": ["종종걸음", "보폭 감소"],
        },
        "injury_risk": {
            "predicted_injury": "fall_risk", "predicted_injury_kr": "낙상 위험", "confidence": 0.85,
            "top3": [
                {"name_kr": "낙상", "probability": 0.85},
                {"name_kr": "발목 염좌", "probability": 0.1},
                {"name_kr": "무릎 통증", "probability": 0.05},
            ],
            "combined_risk_score": 0.85, "combined_risk_grade": "고위험", "timeline": "즉각적인 주의 필요",
        },
        "reasoning": {
            "final_prediction": "parkinsonian", "final_prediction_kr": "파킨슨성 보행", "confidence": 0.92,
            "reasoning_trace": [
                {"step": 1, "label": "IMU 분석", "prediction": "abnormal", "prediction_kr": "비정상", "probability": 0.95},
                {"step": 2, "label": "족저압 분석", "prediction": "asymmetric", "prediction_kr": "비대칭", "probability": 0.88},
                {"step": 3, "label": "종합 추론", "prediction": "parkinsonian", "prediction_kr": "파킨슨", "probability": 0.92},
            ],
            "anomaly_findings": [{"modality": "imu", "anomalies": [{"type": "tremor", "score": 0.8}]}],
            "uncertainty": 0.08, "evidence_strength": 0.92, "is_demo_mode": True,
        },
    },
    "stroke": {
        "classify": {
            "prediction": "ataxic", "prediction_kr": "운동실조", "confidence": 0.87,
            "class_probabilities": {"normal": 0.04, "antalgic": 0.05, "ataxic": 0.87, "parkinsonian": 0.04},
            "is_demo_mode": True,
        },
        "disease_risk": {
            "top_diseases": [{"disease": "stroke", "disease_kr": "뇌졸중", "risk_score": 0.82, "severity": "고위험", "key_signs": ["편측 보행 이상", "발 끌림"], "referral": "신경과"}],
            "ml_prediction": "stroke", "ml_prediction_kr": "뇌졸중", "ml_confidence": 0.82,
            "ml_top3": [
                {"name_kr": "뇌졸중", "probability": 0.82},
                {"name_kr": "소뇌 실조증", "probability": 0.12},
                {"name_kr": "정상", "probability": 0.06},
            ],
            "abnormal_biomarkers": ["편측 보행 이상", "발 끌림"],
        },
        "injury_risk": {
            "predicted_injury": "fall_risk", "predicted_injury_kr": "낙상 위험", "confidence": 0.78,
            "top3": [
                {"name_kr": "낙상", "probability": 0.78},
                {"name_kr": "발목 염좌", "probability": 0.15},
                {"name_kr": "무릎 통증", "probability": 0.07},
            ],
            "combined_risk_score": 0.78, "combined_risk_grade": "고위험", "timeline": "즉각적인 주의 필요",
        },
        "reasoning": {
            "final_prediction": "ataxic", "final_prediction_kr": "운동실조 보행", "confidence": 0.87,
            "reasoning_trace": [
                {"step": 1, "label": "IMU 분석", "prediction": "abnormal", "prediction_kr": "비정상", "probability": 0.89},
                {"step": 2, "label": "족저압 분석", "prediction": "asymmetric", "prediction_kr": "비대칭", "probability": 0.84},
                {"step": 3, "label": "종합 추론", "prediction": "ataxic", "prediction_kr": "운동실조", "probability": 0.87},
            ],
            "anomaly_findings": [{"modality": "skeleton", "anomalies": [{"type": "asymmetry", "score": 0.82}]}],
            "uncertainty": 0.13, "evidence_strength": 0.87, "is_demo_mode": True,
        },
    },
    "fall_risk": {
        "classify": {
            "prediction": "ataxic", "prediction_kr": "운동실조", "confidence": 0.79,
            "class_probabilities": {"normal": 0.06, "antalgic": 0.10, "ataxic": 0.79, "parkinsonian": 0.05},
            "is_demo_mode": True,
        },
        "disease_risk": {
            "top_diseases": [{"disease": "fall_risk", "disease_kr": "낙상 고위험", "risk_score": 0.92, "severity": "매우 고위험", "key_signs": ["균형 불안정", "보행 속도 저하"], "referral": "재활의학과"}],
            "ml_prediction": "fall_risk", "ml_prediction_kr": "낙상 고위험", "ml_confidence": 0.92,
            "ml_top3": [
                {"name_kr": "낙상 고위험", "probability": 0.92},
                {"name_kr": "소뇌 실조증", "probability": 0.06},
                {"name_kr": "정상", "probability": 0.02},
            ],
            "abnormal_biomarkers": ["균형 불안정", "보행 속도 저하"],
        },
        "injury_risk": {
            "predicted_injury": "fall", "predicted_injury_kr": "낙상", "confidence": 0.92,
            "top3": [
                {"name_kr": "낙상", "probability": 0.92},
                {"name_kr": "고관절 골절", "probability": 0.06},
                {"name_kr": "발목 염좌", "probability": 0.02},
            ],
            "combined_risk_score": 0.92, "combined_risk_grade": "매우 고위험", "timeline": "즉각적인 개입 필요",
        },
        "reasoning": {
            "final_prediction": "fall_risk", "final_prediction_kr": "낙상 고위험", "confidence": 0.92,
            "reasoning_trace": [
                {"step": 1, "label": "IMU 분석", "prediction": "unstable", "prediction_kr": "불안정", "probability": 0.91},
                {"step": 2, "label": "족저압 분석", "prediction": "unbalanced", "prediction_kr": "불균형", "probability": 0.90},
                {"step": 3, "label": "종합 추론", "prediction": "fall_risk", "prediction_kr": "낙상 위험", "probability": 0.92},
            ],
            "anomaly_findings": [{"modality": "imu", "anomalies": [{"type": "instability", "score": 0.91}]}],
            "uncertainty": 0.08, "evidence_strength": 0.92, "is_demo_mode": True,
        },
    },
}

_FALLBACK_REPORT = {
    "normal": "현재 보행 패턴은 정상 범위 내에 있습니다. 규칙적인 운동과 주기적인 보행 모니터링을 권장합니다.",
    "parkinsons": "파킨슨성 보행 패턴이 강하게 의심됩니다. 신경과 전문의 진료를 권장합니다.",
    "stroke": "뇌졸중 후유증으로 인한 보행 이상이 감지되었습니다. 신경과 및 재활의학과 진료를 권장합니다.",
    "fall_risk": "낙상 위험이 매우 높습니다. 즉각적인 보행 보조기구 사용과 전문의 상담을 권장합니다.",
}

_FALLBACK_ACTIONS = {
    "normal": "- 규칙적인 유산소 운동 유지\n- 월 1회 보행 측정 권장\n- 균형 강화 운동 추가",
    "parkinsons": "- 보행 보조기구 사용 권장\n- 보호자 동반 보행\n- 즉각적인 신경과 검진",
    "stroke": "- 재활 치료 프로그램 참여\n- 보행 보조기구 사용\n- 신경과·재활의학과 협진",
    "fall_risk": "- 즉시 보행 보조기구 사용\n- 가정 내 낙상 위험 환경 제거\n- 재활의학과 즉시 방문",
}


async def _call_ollama(prompt: str, fallback: str) -> str:
    try:
        response = await _ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content.strip()
    except Exception as e:
        logger.warning("Ollama 호출 실패 (model=%s): %s", OLLAMA_MODEL, e)
        return fallback


@router.get("/sample")
async def get_sample(gait_profile: str = "normal"):
    imu = np.zeros((128, 6)).tolist()
    pressure = np.zeros((16, 8)).tolist()
    skeleton = np.zeros((128, 17, 3)).tolist()

    return {
        "gait_profile": gait_profile,
        "sensor_data": {"imu": imu, "pressure": pressure, "skeleton": skeleton},
        "features": {"cadence": 105.0, "stride_length": 1.2},
    }


@router.post("/analyze")
async def analyze(payload: dict):
    gait_profile = payload.get("gait_profile", "parkinsons")
    ctx = _PROFILE_CONTEXT.get(gait_profile, _PROFILE_CONTEXT["parkinsons"])
    mock = _PROFILE_RESPONSES.get(gait_profile, _PROFILE_RESPONSES["parkinsons"])

    disease = ctx["disease"]
    abnormal_biomarkers = ctx["abnormal_biomarkers"]
    fall_risk = ctx["fall_risk"]

    prompt_report = f"""당신은 세계 최고의 보행 분석 AI 전문가입니다.
다음 보행 분석 결과를 바탕으로 환자에게 제공할 '종합 소견서'를 3문장 이내의 자연스러운 한국어로 작성하세요.

분석 결과:
- 주요 의심 질환: {disease}
- 비정상 생체 지표: {abnormal_biomarkers}
- 낙상 위험도: {fall_risk}

출력은 오직 소견서 텍스트만 포함하세요."""

    prompt_actions = f"""당신은 재활 치료사입니다.
다음 보행 분석 결과를 바탕으로 환자가 즉각적으로 취해야 할 '우선 권장 행동' 3가지를 한국어로 작성하세요.

분석 결과:
- 주요 의심 질환: {disease}
- 낙상 위험도: {fall_risk}

출력은 '-'로 시작하는 목록 형태로 3줄만 작성하세요. 다른 말은 절대 추가하지 마세요."""

    report_kr, actions_text = await asyncio.gather(
        _call_ollama(prompt_report, _FALLBACK_REPORT.get(gait_profile, _FALLBACK_REPORT["parkinsons"])),
        _call_ollama(prompt_actions, _FALLBACK_ACTIONS.get(gait_profile, _FALLBACK_ACTIONS["parkinsons"])),
    )

    priority_actions = [line.lstrip("- ").strip() for line in actions_text.splitlines() if line.strip()]
    if not priority_actions:
        priority_actions = _FALLBACK_ACTIONS.get(gait_profile, "").replace("- ", "").splitlines()

    return {
        **mock,
        "injury_risk": {**mock["injury_risk"], "priority_actions": priority_actions},
        "reasoning": {**mock["reasoning"], "report_kr": report_kr},
    }
