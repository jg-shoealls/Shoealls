"""BioMistral / BioBERT 기반 임상 보행 분석 보고서 생성기.

사용 우선순위:
  1. BioMistral-7B (생성형 LLM) — GPU 8GB+ 권장
  2. BioBERT fill-mask (경량, CPU 가능)
  3. 구조화 템플릿 fallback (인터넷/GPU 없는 환경)

config (configs/default.yaml hf_encoders.llm_report):
  enabled        : true/false
  model_id       : "BioMistral/BioMistral-7B"
  fallback_model_id: "dmis-lab/biobert-base-cased-v1.2"
  max_new_tokens : 512
  temperature    : 0.3
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────

@dataclass
class GaitSummary:
    """LLM 보고서 생성에 필요한 보행 분석 요약 정보."""
    predicted_class: str          # "normal" | "antalgic" | "ataxic" | "parkinsonian"
    predicted_class_kr: str
    confidence: float
    disease_risks: dict[str, float]   # 질환명 → 위험도(0-1)
    injury_risks: dict[str, float]    # 부위명 → 위험도(0-1)
    gait_features: dict[str, float]   # feature 이름 → 값
    session_id: Optional[str] = None

    @property
    def top_disease(self) -> tuple[str, float]:
        if not self.disease_risks:
            return ("없음", 0.0)
        top = max(self.disease_risks.items(), key=lambda x: x[1])
        return top

    @property
    def top_injury(self) -> tuple[str, float]:
        if not self.injury_risks:
            return ("없음", 0.0)
        top = max(self.injury_risks.items(), key=lambda x: x[1])
        return top


@dataclass
class LLMReport:
    """생성된 임상 보고서."""
    summary_kr: str
    clinical_findings_kr: str
    recommendations_kr: str
    risk_level: str          # "낮음" | "중간" | "높음"
    generated_by: str        # "BioMistral" | "BioBERT" | "template"

    @property
    def full_report_kr(self) -> str:
        divider = "─" * 50
        return (
            f"[보행 AI 임상 분석 보고서]\n{divider}\n"
            f"【요약】\n{self.summary_kr}\n\n"
            f"【임상 소견】\n{self.clinical_findings_kr}\n\n"
            f"【권고 사항】\n{self.recommendations_kr}\n\n"
            f"위험 등급: {self.risk_level}  |  생성 엔진: {self.generated_by}\n"
            f"{divider}"
        )


# ── 메인 생성기 ────────────────────────────────────────────────────────────────

class LLMReportGenerator:
    """BioMistral/BioBERT 기반 임상 보고서 생성기.

    인스턴스화 시 available backend 를 자동 선택한다:
      1. BioMistral-7B   (GPU 필요, 고품질 생성)
      2. Template-based  (항상 동작, 오프라인 가능)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = (config or {}).get("hf_encoders", {}).get("llm_report", {})
        self.model_id = cfg.get("model_id", "BioMistral/BioMistral-7B")
        self.fallback_id = cfg.get("fallback_model_id", "dmis-lab/biobert-base-cased-v1.2")
        self.max_new_tokens = cfg.get("max_new_tokens", 512)
        self.temperature = cfg.get("temperature", 0.3)

        self._pipeline = None
        self._backend = "template"
        self._try_load_pipeline()

    def _try_load_pipeline(self) -> None:
        try:
            import torch
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

            device_id = 0 if __import__("torch").cuda.is_available() else -1

            # BioMistral-7B 시도
            try:
                logger.info(f"BioMistral 로드 시도: {self.model_id}")
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=__import__("torch").float16 if device_id >= 0 else __import__("torch").float32,
                    device_map="auto" if device_id >= 0 else None,
                )
                self._pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=device_id,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                )
                self._backend = "BioMistral"
                logger.info("BioMistral-7B 로드 성공")
                return
            except Exception as e:
                logger.warning(f"BioMistral 로드 실패: {e}")

            logger.info("Template fallback 사용")

        except ImportError:
            logger.warning("transformers 미설치 — template fallback 사용")

    def generate(self, summary: GaitSummary) -> LLMReport:
        """보행 요약 정보를 받아 임상 보고서를 생성한다."""
        if self._backend == "BioMistral" and self._pipeline is not None:
            return self._generate_with_llm(summary)
        return self._generate_from_template(summary)

    # ── LLM 경로 ──────────────────────────────────────────────────────────

    def _build_prompt(self, summary: GaitSummary) -> str:
        top_disease, d_risk = summary.top_disease
        top_injury, i_risk = summary.top_injury
        feats = summary.gait_features

        speed   = feats.get("gait_speed", 1.2)
        cadence = feats.get("cadence", 115)
        sway    = feats.get("cop_sway", 0.04)
        symm    = feats.get("step_symmetry", 0.92)

        return textwrap.dedent(f"""
            You are a clinical gait analysis AI assistant. Write a concise Korean clinical report.

            Patient gait data:
            - Gait class: {summary.predicted_class_kr} (confidence {summary.confidence:.1%})
            - Gait speed: {speed:.2f} m/s, Cadence: {cadence:.0f} steps/min
            - Step symmetry: {symm:.2f}, CoP sway: {sway:.3f}
            - Highest disease risk: {top_disease} ({d_risk:.1%})
            - Highest injury risk: {top_injury} ({i_risk:.1%})

            Write a Korean clinical gait report with three sections:
            1. 요약 (2 sentences)
            2. 임상 소견 (3-4 sentences)
            3. 권고 사항 (2-3 bullet points)

            Report:
        """).strip()

    def _generate_with_llm(self, summary: GaitSummary) -> LLMReport:
        prompt = self._build_prompt(summary)
        try:
            result = self._pipeline(prompt)[0]["generated_text"]
            # 프롬프트 이후 부분만 추출
            generated = result[len(prompt):].strip()
            sections = self._parse_llm_output(generated)
            return LLMReport(
                summary_kr=sections.get("요약", self._template_summary(summary)),
                clinical_findings_kr=sections.get("임상 소견", self._template_findings(summary)),
                recommendations_kr=sections.get("권고 사항", self._template_recommendations(summary)),
                risk_level=self._risk_level(summary),
                generated_by="BioMistral",
            )
        except Exception as e:
            logger.warning(f"BioMistral 생성 실패 ({e}), template 사용")
            return self._generate_from_template(summary)

    @staticmethod
    def _parse_llm_output(text: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        current_key = None
        current_lines: list[str] = []
        for line in text.splitlines():
            for key in ["요약", "임상 소견", "권고 사항"]:
                if key in line:
                    if current_key:
                        sections[current_key] = "\n".join(current_lines).strip()
                    current_key = key
                    current_lines = []
                    break
            else:
                if current_key:
                    current_lines.append(line)
        if current_key:
            sections[current_key] = "\n".join(current_lines).strip()
        return sections

    # ── Template 경로 ──────────────────────────────────────────────────────

    _CLASS_DESC = {
        "normal":       "정상 범위의 보행 패턴이 관찰되었습니다.",
        "antalgic":     "통증 회피성 보행(절뚝거림)이 감지되어 하지 통증 또는 부상을 시사합니다.",
        "ataxic":       "운동실조성 보행이 감지되어 소뇌 또는 전정 기관 이상을 시사합니다.",
        "parkinsonian": "파킨슨형 보행 패턴이 감지되어 기저핵 기능 이상을 시사합니다.",
    }

    _RISK_THRESHOLDS = {"낮음": 0.35, "중간": 0.60, "높음": 1.01}

    def _risk_level(self, summary: GaitSummary) -> str:
        _, d_risk = summary.top_disease
        _, i_risk = summary.top_injury
        max_risk = max(d_risk, i_risk)
        for label, threshold in self._RISK_THRESHOLDS.items():
            if max_risk < threshold:
                return label
        return "높음"

    def _template_summary(self, summary: GaitSummary) -> str:
        cls_desc = self._CLASS_DESC.get(summary.predicted_class, "비정상 보행 패턴이 감지되었습니다.")
        top_d, d_risk = summary.top_disease
        return (
            f"보행 AI 분석 결과 {summary.predicted_class_kr}으로 분류되었습니다 "
            f"(신뢰도 {summary.confidence:.1%}). "
            f"{cls_desc} "
            f"가장 높은 질환 위험도는 {top_d}({d_risk:.1%})입니다."
        )

    def _template_findings(self, summary: GaitSummary) -> str:
        feats = summary.gait_features
        speed   = feats.get("gait_speed", 1.2)
        cadence = feats.get("cadence", 115.0)
        symm    = feats.get("step_symmetry", 0.92)
        sway    = feats.get("cop_sway", 0.04)
        heel    = feats.get("heel_pressure_ratio", 0.33)

        findings = [
            f"보행 속도 {speed:.2f} m/s, 보행 리듬 {cadence:.0f} steps/min"
            + (" (정상 범위)" if 0.9 <= speed <= 1.5 and 100 <= cadence <= 130 else " (비정상)"),
            f"좌우 보행 대칭도 {symm:.2f}" + (" (양호)" if symm >= 0.85 else " (불균형 감지)"),
            f"무게중심 흔들림(CoP Sway) {sway:.3f}" + (" (안정)" if sway < 0.1 else " (불안정)"),
            f"뒤꿈치 접지 압력 비율 {heel:.1%}" + (" (과도한 뒤꿈치 착지)" if heel > 0.45 else ""),
        ]
        return "\n".join(f"• {f}" for f in findings if f.endswith(")") or not f.endswith("()"))

    def _template_recommendations(self, summary: GaitSummary) -> str:
        recs: list[str] = []
        _, d_risk = summary.top_disease
        _, i_risk = summary.top_injury
        top_d, _ = summary.top_disease
        top_i, _ = summary.top_injury

        if d_risk >= 0.6:
            recs.append(f"{top_d} 정밀 검사 권고 (신경과/정형외과 협진)")
        if i_risk >= 0.5:
            recs.append(f"{top_i} 예방을 위한 족부 물리치료 권고")

        feats = summary.gait_features
        if feats.get("step_symmetry", 1.0) < 0.80:
            recs.append("좌우 대칭 보행 훈련 및 균형 재활 프로그램 적용 권고")
        if feats.get("gait_speed", 1.2) < 0.8:
            recs.append("보행 속도 개선을 위한 근력 강화 운동 권고")
        if feats.get("cop_sway", 0.0) > 0.15:
            recs.append("낙상 위험 관리: 보행 보조기 사용 검토")

        if not recs:
            recs.append("현재 보행 상태는 양호합니다. 3개월 후 재평가를 권장합니다.")

        return "\n".join(f"• {r}" for r in recs)

    def _generate_from_template(self, summary: GaitSummary) -> LLMReport:
        return LLMReport(
            summary_kr=self._template_summary(summary),
            clinical_findings_kr=self._template_findings(summary),
            recommendations_kr=self._template_recommendations(summary),
            risk_level=self._risk_level(summary),
            generated_by="template",
        )
