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
        return f"""당신은 숙련된 보행 분석 AI 전문가입니다. 
다음의 정밀 분석 데이터를 바탕으로 환자가 이해하기 쉬운 '종합 건강 리포트'를 작성해주세요.

[분석 요약]
- 판정 결과: {data['prediction']} (확신도: {data['confidence']:.1%})
- 불확실성: {data['uncertainty']:.1%}
- 종합 근거 강도: {data['evidence_strength']:.1%}

[이상 징후 상세]
{data['anomalies']}

[감별 진단 근거]
- 찬성 근거 강도: {data['pro_scores']}
- 반대 근거 강도: {data['con_scores']}

작성 지침:
1. 전문 용어보다는 환자가 이해하기 쉬운 비유와 설명을 사용하세요.
2. 현재 상태의 심각성과 개선을 위한 긍정적인 조언을 포함하세요.
3. 3~4개 문단으로 구성된 자연스러운 한국어로 작성하세요.
"""

    def _build_clinical_prompt(self, data: Dict) -> str:
        return f"""당신은 대학병원 신경과/정형외과 전문의를 보조하는 의료 AI 컨설턴트입니다.
다음의 멀티모달 보행 분석 데이터를 기반으로 심층적인 '임상 분석 소견서'를 작성하세요.

[데이터 요약]
- 가설 진단: {data['prediction']} (Softmax Probs: {data['probabilities']})
- 신뢰도 보정 결과: {data['confidence']:.1%} (Uncertainty: {data['uncertainty']:.1%})
- 교차 모달 지지도: {data['cross_support']}

[모달리티별 이상 패턴 점수]
{data['anomalies_raw']}

[추론 체인 과정]
{data['reasoning_trace']}

작성 지침:
1. 의학적 관점에서 보행 특성(Gait Pathomechanics)과 질환 간의 연관성을 심층 분석하세요.
2. 각 모달리티(IMU, 족저압, 스켈레톤) 간의 불일치나 특이사항이 있다면 지적하세요.
3. 감별 진단을 위한 추가 검사 제안이나 임상적 주의사항을 포함하세요.
4. 전문적인 의학 용어를 사용하여 한국어로 작성하세요.
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
