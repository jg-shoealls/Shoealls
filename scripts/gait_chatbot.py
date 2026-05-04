import torch
import yaml
import json
import requests
import os
from src.models.reasoning_engine import GaitReasoningEngine

class GaitChatbot:
    def __init__(self, config_path="configs/default.yaml"):
        # 1. 분석 엔진 준비
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.engine = GaitReasoningEngine(self.config).to(self.device)
        self.engine.eval()
        
        self.ollama_url = "http://127.0.0.1:11434/api/chat"
        self.model = "llama3"
        self.history = [] # 대화 기록 저장
        
        # 시스템 프롬프트 설정 (페르소나 부여)
        self.system_prompt = """You must always respond in Korean (한국어). Never use English in your responses.

당신은 'ShoeAlls' 스마트 인솔 기반의 전문 보행 분석 상담 AI입니다.
제공된 분석 결과를 바탕으로 사용자의 질문에 친절하고 전문적으로 답해야 합니다.

[상담 지침]
1. 모든 답변은 반드시 한국어로만 작성하세요. 영어 사용 금지.
2. 사용자가 자신의 상태를 쉽게 이해할 수 있도록 비유를 섞어 설명하세요.
3. 분석 결과에 없는 내용을 지어내지 마세요.
4. 의학적 진단은 '가능성'으로 언급하며, 최종 판단은 반드시 전문의와 상의하라고 조언하세요.
5. 재활 운동이나 생활 습관 개선 등 실질적인 조언을 포함하세요.
"""

    def analyze_current_gait(self):
        """실시간(가상) 보행 데이터를 분석하여 구조화 리포트 반환"""
        batch = {
            "imu": torch.randn(1, 6, 128).to(self.device),
            "pressure": torch.randn(1, 128, 1, 16, 8).to(self.device),
            "skeleton": torch.randn(1, 3, 128, 17).to(self.device),
        }
        with torch.no_grad():
            result = self.engine.reason(batch)
            report = self.engine.explain(result, sample_idx=0)
        return report

    def generate_report(self, structured_report: str) -> str:
        """구조화 분석 리포트 → llama3가 작성한 임상 리포트"""
        prompt = f"""You must respond in Korean only. Never use English.

아래는 보행 분석 AI가 생성한 구조화 데이터입니다.
이 데이터를 바탕으로 환자가 읽기 쉬운 임상 보행 리포트를 한국어로 작성해주세요.

리포트 형식:
1. 전체 요약 (2~3문장, 일반인이 이해할 수 있는 언어)
2. 주요 발견사항 (감지된 이상 패턴을 쉽게 설명)
3. 의학적 소견 (가능성 수준으로 언급, 단정 금지)
4. 권장 사항 (재활 운동, 생활 습관, 전문의 상담 여부)

[분석 데이터]
{structured_report}
"""
        messages = [{"role": "user", "content": prompt}]
        data = {"model": self.model, "messages": messages, "stream": True}

        print("\n" + "=" * 60)
        print("  AI 임상 보행 리포트")
        print("=" * 60 + "\n")

        report_text = ""
        try:
            with requests.post(self.ollama_url, json=data, stream=True, timeout=300) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        print(token, end="", flush=True)
                        report_text += token
                        if chunk.get("done"):
                            break
            print("\n" + "=" * 60)
        except Exception as e:
            print(f"오류: {str(e)}")
        return report_text

    def chat(self, user_input, analysis_context):
        """사용자 질문에 답변 생성"""
        
        # 대화 기록에 시스템 컨텍스트와 분석 결과 추가 (첫 대화 시)
        if not self.history:
            self.history.append({
                "role": "system", 
                "content": f"{self.system_prompt}\n\n[현재 사용자의 보행 분석 데이터]\n{analysis_context}"
            })

        self.history.append({"role": "user", "content": user_input})

        data = {
            "model": self.model,
            "messages": self.history,
            "stream": True
        }

        try:
            answer = ""
            with requests.post(self.ollama_url, json=data, stream=True, timeout=300) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        print(token, end="", flush=True)
                        answer += token
                        if chunk.get("done"):
                            break
            print()
            self.history.append({"role": "assistant", "content": answer})
            return ""
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"

def run_chat_demo():
    print("="*60)
    print("  ShoeAlls AI 보행 분석 상담 서비스 (Powered by Llama 3)")
    print("="*60)

    bot = GaitChatbot()

    print("\n[1단계] 보행 데이터 분석 중...")
    context = bot.analyze_current_gait()

    print("\n[2단계] AI 임상 리포트 생성 중...")
    bot.generate_report(context)

    print("\n[3단계] 상담을 시작합니다. 궁금한 점을 물어보세요!")
    print("(종료하려면 '나가기' 또는 'exit'를 입력하세요.)")

    while True:
        user_msg = input("\n나: ")
        if user_msg.lower() in ["exit", "quit", "나가기", "종료"]:
            print("\n상담을 종료합니다. 건강한 하루 되세요!")
            break
            
        print("\nAI 상담사: ", end="", flush=True)
        answer = bot.chat(user_msg, context)
        print(answer)

if __name__ == "__main__":
    run_chat_demo()
