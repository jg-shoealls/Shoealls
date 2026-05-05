import logging
import os
import asyncio
from typing import Optional, List, Dict
import ollama

logger = logging.getLogger(__name__)

class LLMService:
    """Ollama 기반 LLM 서비스 싱글톤."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.client = ollama.AsyncClient()
        self._initialized = True
        logger.info("LLMService initialized with model: %s", self.model)

    async def generate_report(self, structured_data: Dict) -> str:
        """분석 데이터를 바탕으로 자연어 리포트 생성."""
        prompt = self._build_report_prompt(structured_data)
        return await self._call_ollama(prompt)

    async def generate_clinical_notes(self, structured_data: Dict) -> str:
        """전문의를 위한 심층 임상 소견서 생성."""
        prompt = self._build_clinical_prompt(structured_data)
        return await self._call_ollama(prompt)

    def _build_report_prompt(self, data: Dict) -> str:
        return f"""당신은 숙련된 보행 분석 AI 전문가이자 친절한 건강 상담사입니다. 
다음의 정밀 분석 데이터를 바탕으로 환자가 이해하기 쉬운 'AI 보행 건강 리포트'를 작성해주세요.

### [분석 요약]
- **최종 판정**: {data['prediction']}
- **판정 신뢰도**: {data['confidence']:.1%}
- **데이터 분석 강도**: {data['evidence_strength']:.1%}

### [상세 이상 징후]
{data['anomalies']}

### [감별 진단 근거]
- **주요 일치 지표**: {data['pro_scores']}
- **상충 가능 지표**: {data['con_scores']}

---
**작성 지침:**
1. **친절한 어조**: 환자가 불안해하지 않도록 친절하고 전문적인 어조를 유지하세요.
2. **쉬운 설명**: '비대칭성', '운동실조' 등의 용어를 사용할 때는 "걷는 모양이 좌우가 조금 다르시네요"와 같이 쉽게 풀어서 설명해주세요.
3. **구조적 구성**:
   - 1문단: 현재 보행 상태에 대한 전체적인 총평
   - 2문단: 발견된 주요 이상 징후와 그 의미 설명
   - 3문단: 건강 개선을 위한 일상생활 속 실천 제안 (예: 근력 운동, 보행 주의사항)
4. **언어**: 반드시 자연스러운 한국어로 작성하세요.
"""

    def _build_clinical_prompt(self, data: Dict) -> str:
        return f"""당신은 대학병원 신경과 및 정형외과 전문의를 보조하는 '의료 AI 임상 컨설턴트'입니다.
다음의 멀티모달 보행 데이터(IMU, 족저압, 스켈레톤) 분석 결과를 기반으로 전문적인 '임상 분석 소견서(Clinical Assessment)'를 작성하세요.

### [Data Synthesis]
- **Primary Hypothesis**: {data['prediction']} (Confidence: {data['confidence']:.1%})
- **Softmax Distribution**: {data['probabilities']}
- **Reasoning Calibration**: Uncertainty Index {data['uncertainty']:.1%}, Cross-Modal Support {data['cross_support']}

### [Modal-specific Abnormalities]
{data['anomalies_raw']}

### [Chain-of-Reasoning Trace]
{data['reasoning_trace']}

---
**작성 지침:**
1. **의학적 심층 분석**: 보행 병태생리학(Gait Pathophysiology)적 관점에서 특징을 분석하세요. (예: 파킨슨 보행의 경우 서동증, 소보행, 동결 현상 등과 연결)
2. **모달리티 통합 해석**: IMU의 가속도 변화, 족저압의 압심점(COP) 이동, 스켈레톤의 관절 각도 변화가 어떻게 상호 보완적으로 특정 질환을 지지하는지 설명하세요.
3. **임상적 권고**: 차등 진단(Differential Diagnosis)을 위해 필요한 추가 임상 검사(예: UPDRS score, MRI, EMG)나 처방 방향에 대한 조언을 포함하세요.
4. **전문 용어 사용**: 의료 현장에서 사용하는 전문적인 의학 용어를 사용하여 격식 있는 한국어로 작성하세요.
"""

    async def _call_ollama(self, prompt: str) -> str:
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.message.content.strip()
        except Exception as e:
            logger.error("Ollama API 호출 중 오류 발생: %s", e)
            return "LLM 리포트를 생성할 수 없습니다. (Ollama 서비스 연결 확인 필요)"

# 싱글톤 인스턴스 노출
llm_service = LLMService()
