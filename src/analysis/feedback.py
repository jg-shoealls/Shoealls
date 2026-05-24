"""Corrective feedback generator: personalized Korean gait improvement recommendations."""

import numpy as np
from dataclasses import dataclass

from .foot_zones import FootAnalysisResult
from .gait_profile import PersonalGaitProfiler, DeviationReport, GaitBaseline
from .injury_risk import InjuryRiskReport, InjuryRisk


@dataclass
class FeedbackItem:
    """Single feedback recommendation."""
    category: str        # exercise / footwear / posture / medical
    priority: int        # 1=highest
    title: str
    description: str
    exercises: list[str]


@dataclass
class PersonalizedFeedback:
    """Complete personalized feedback report."""
    items: list[FeedbackItem]
    overall_status: str
    encouragement: str
    report_kr: str


class CorrektiveFeedbackGenerator:
    """Generates personalized Korean gait improvement recommendations.

    Combines injury risk assessment and personal deviation data to produce
    actionable, prioritized feedback.
    """

    def generate(
        self,
        injury_report: InjuryRiskReport,
        deviation_report: DeviationReport | None = None,
        baseline: GaitBaseline | None = None,
    ) -> PersonalizedFeedback:
        """Generate personalized feedback.

        Args:
            injury_report: Current injury risk assessment.
            deviation_report: Optional deviation from personal baseline.
            baseline: Optional personal baseline data.
        """
        items = []
        priority = 1

        # 1. Injury-based recommendations
        high_risks = [r for r in injury_report.risks if r.risk_score >= 0.5]
        moderate_risks = [r for r in injury_report.risks if 0.25 <= r.risk_score < 0.5]

        for risk in sorted(high_risks, key=lambda x: -x.risk_score):
            item = self._injury_to_feedback(risk, priority)
            items.append(item)
            priority += 1

        for risk in sorted(moderate_risks, key=lambda x: -x.risk_score):
            item = self._injury_to_feedback(risk, priority)
            items.append(item)
            priority += 1

        # 2. Deviation-based recommendations
        if deviation_report and deviation_report.alerts:
            for alert in deviation_report.alerts:
                item = self._deviation_to_feedback(alert, priority)
                if item:
                    items.append(item)
                    priority += 1

        # 3. General gait improvement tips based on baseline
        if baseline and baseline.num_sessions >= 1:
            general = self._baseline_tips(baseline, priority)
            items.extend(general)

        # Overall status
        if injury_report.overall_risk >= 0.75:
            status = "주의 필요"
            encouragement = "부상 위험이 높습니다. 아래 권장사항을 꼭 확인해주세요."
        elif injury_report.overall_risk >= 0.5:
            status = "개선 권장"
            encouragement = "몇 가지 개선이 필요하지만, 꾸준한 관리로 충분히 좋아질 수 있습니다!"
        elif injury_report.overall_risk >= 0.25:
            status = "양호"
            encouragement = "전체적으로 좋은 상태입니다. 작은 습관 개선으로 더 좋아질 수 있어요."
        else:
            status = "매우 양호"
            encouragement = "훌륭한 보행 패턴을 유지하고 있습니다! 계속 이대로 유지하세요."

        report = self._build_report(items, status, encouragement, injury_report, deviation_report)

        return PersonalizedFeedback(
            items=items,
            overall_status=status,
            encouragement=encouragement,
            report_kr=report,
        )

    def _injury_to_feedback(self, risk: InjuryRisk, priority: int) -> FeedbackItem:
        """Convert injury risk to actionable feedback."""
        exercises_map = {
            "plantar_fasciitis": [
                "발바닥 근막 스트레칭 (벽에 발끝 대고 30초 유지, 3세트)",
                "발가락 수건 잡기 운동 (수건을 발가락으로 당기기, 10회 3세트)",
                "종아리 스트레칭 (계단 끝에 서서 뒤꿈치 내리기, 30초 3세트)",
                "얼음 마사지 (얼린 물병 위에서 발바닥 굴리기, 5분)",
            ],
            "metatarsal_stress": [
                "발가락 벌리기 운동 (고무밴드 이용, 10회 3세트)",
                "앞발 체중 분산 훈련 (맨발로 균등 체중 싣기 연습)",
                "앞발 스트레칭 (무릎 꿇고 앉아 발등 늘리기, 30초)",
                "부드러운 지면에서 가벼운 걷기 (잔디밭 등, 15분)",
            ],
            "ankle_sprain": [
                "발목 알파벳 운동 (발끝으로 알파벳 쓰기)",
                "한 발 서기 균형 훈련 (30초씩 좌우 번갈아, 5세트)",
                "발목 밴드 운동 (내번/외번 저항 운동, 15회 3세트)",
                "보수(BOSU) 볼 위 균형 잡기 (30초 3세트)",
            ],
            "heel_spur": [
                "뒤꿈치 쿠션 패드 삽입",
                "종아리-아킬레스건 스트레칭 (벽 밀기, 30초 3세트)",
                "발뒤꿈치 들기 운동 (천천히 올리고 내리기, 15회 3세트)",
                "딱딱한 바닥 위 장시간 서기 자제",
            ],
            "flat_foot": [
                "아치 강화 운동 (숏풋 운동: 발바닥 오므리기, 10초 유지 10회)",
                "발가락 끝으로 걷기 (30초씩 3세트)",
                "골프공 발바닥 굴리기 (3분, 아치 부분 집중)",
                "아치 지지 인솔 착용 권장",
            ],
            "high_arch": [
                "발바닥 전체 스트레칭 (발등 늘리기, 30초 유지)",
                "종아리 폼롤러 마사지 (앞뒤로 굴리기, 2분)",
                "발가락 스프레드 운동 (발가락 벌리고 모으기, 20회)",
                "쿠션이 좋은 중립 타입 신발 착용",
            ],
        }

        category_map = {
            "plantar_fasciitis": "exercise",
            "metatarsal_stress": "exercise",
            "ankle_sprain": "exercise",
            "heel_spur": "footwear",
            "flat_foot": "footwear",
            "high_arch": "footwear",
        }

        return FeedbackItem(
            category=category_map.get(risk.name, "exercise"),
            priority=priority,
            title=f"{risk.korean_name} 예방 및 관리",
            description=f"{risk.severity} 수준 - {', '.join(risk.contributing_factors)}. {risk.recommendation}",
            exercises=exercises_map.get(risk.name, []),
        )

    def _deviation_to_feedback(self, alert: dict, priority: int) -> FeedbackItem | None:
        """Convert a deviation alert to feedback."""
        metric = alert["metric"]
        severity = alert["severity"]

        feedback_map = {
            "cop_sway": FeedbackItem(
                category="exercise",
                priority=priority,
                title="균형 능력 저하 감지",
                description=f"{severity} - 체중심 흔들림이 평소보다 증가했습니다.",
                exercises=[
                    "한 발 서기 연습 (30초씩 좌우, 3세트)",
                    "눈 감고 서기 (안전한 곳에서, 15초씩)",
                    "일직선 걷기 (발뒤꿈치-발끝 일직선, 10걸음)",
                ],
            ),
            "step_symmetry": FeedbackItem(
                category="posture",
                priority=priority,
                title="좌우 보행 비대칭 증가",
                description=f"{severity} - 좌우 보행 패턴의 차이가 커졌습니다.",
                exercises=[
                    "거울 앞에서 보행 자세 확인하며 걷기",
                    "좌우 균등하게 체중 싣기 연습",
                    "약한 쪽 다리 근력 강화 운동",
                ],
            ),
            "cadence": FeedbackItem(
                category="posture",
                priority=priority,
                title="보행 속도 변화 감지",
                description=f"{severity} - 보행 속도(보행률)가 평소와 다릅니다.",
                exercises=[
                    "편안한 속도로 10분 연속 걷기",
                    "메트로놈 앱으로 일정 리듬 유지 연습",
                ],
            ),
            "stride_regularity": FeedbackItem(
                category="posture",
                priority=priority,
                title="보폭 불규칙 증가",
                description=f"{severity} - 보폭의 일관성이 떨어졌습니다.",
                exercises=[
                    "일정 간격 표시된 바닥에서 걷기 연습",
                    "천천히 의식적으로 걷기 (5분)",
                ],
            ),
            "ml_index": FeedbackItem(
                category="posture",
                priority=priority,
                title="좌우 체중 분포 변화",
                description=f"{severity} - 좌우 체중 분포가 평소와 다릅니다.",
                exercises=[
                    "체중계 두 개로 좌우 균형 확인",
                    "양쪽 발에 균등하게 체중 싣기 의식 훈련",
                ],
            ),
        }

        return feedback_map.get(metric)

    def _baseline_tips(self, baseline: GaitBaseline, start_priority: int) -> list[FeedbackItem]:
        """General tips based on baseline patterns."""
        tips = []
        p = start_priority

        # Low stride regularity
        if baseline.stride_regularity[0] < 0.5:
            tips.append(FeedbackItem(
                category="exercise",
                priority=p,
                title="보폭 규칙성 향상 추천",
                description="보폭 규칙성 지수가 낮습니다. 꾸준한 연습으로 개선할 수 있습니다.",
                exercises=[
                    "리듬에 맞춰 걷기 연습 (음악 BPM 100-120)",
                    "트레드밀에서 일정 속도 걷기 (10분)",
                ],
            ))
            p += 1

        # Low step symmetry
        if baseline.step_symmetry[0] < 0.7:
            tips.append(FeedbackItem(
                category="exercise",
                priority=p,
                title="보행 대칭성 향상 추천",
                description="좌우 보행 대칭성을 개선하면 부상 위험을 줄일 수 있습니다.",
                exercises=[
                    "좌우 번갈아 한 발 서기 (30초씩 5세트)",
                    "약한 쪽 다리 스쿼트 (10회 3세트)",
                ],
            ))
            p += 1

        return tips

    def _build_report(
        self,
        items: list[FeedbackItem],
        status: str,
        encouragement: str,
        injury_report: InjuryRiskReport,
        deviation_report: DeviationReport | None,
    ) -> str:
        """Build full Korean report text."""
        lines = [
            "=" * 60,
            "  맞춤형 보행 분석 피드백 리포트",
            "=" * 60,
            "",
            f"  종합 상태: {status}",
            f"  {encouragement}",
            "",
        ]

        # Injury risk summary
        lines.append("─" * 60)
        lines.append("  [부상 위험 평가]")
        lines.append("")
        for risk in sorted(injury_report.risks, key=lambda r: -r.risk_score):
            bar = "█" * int(risk.risk_score * 10) + "░" * (10 - int(risk.risk_score * 10))
            lines.append(f"  {risk.korean_name:12s} [{bar}] {risk.severity}")

        # Deviation summary
        if deviation_report and deviation_report.alerts:
            lines.append("")
            lines.append("─" * 60)
            lines.append("  [개인 기준 대비 변화]")
            lines.append("")
            for alert in deviation_report.alerts:
                lines.append(f"  [{alert['severity']}] {alert['message']}")

        # Recommendations
        if items:
            lines.append("")
            lines.append("─" * 60)
            lines.append("  [맞춤형 운동 및 관리 권장사항]")
            lines.append("")
            for item in items:
                lines.append(f"  ◆ [{item.priority}순위] {item.title}")
                lines.append(f"    {item.description}")
                if item.exercises:
                    lines.append("    추천 운동:")
                    for ex in item.exercises:
                        lines.append(f"      • {ex}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
