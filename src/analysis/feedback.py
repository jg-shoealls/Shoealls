"""Corrective feedback generator: personalized Korean gait improvement recommendations."""

import numpy as np
from dataclasses import dataclass

from .foot_zones import FootAnalysisResult
from .gait_profile import PersonalGaitProfiler, DeviationReport, GaitBaseline
from .injury_risk import InjuryRiskReport, InjuryRisk
from .prodromal_biomarkers import ProdromalPanel


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
        prodromal_panel: ProdromalPanel | None = None,
        feedback_context: dict | None = None,
    ) -> PersonalizedFeedback:
        """Generate personalized feedback.

        Args:
            injury_report: Current injury risk assessment.
            deviation_report: Optional deviation from personal baseline.
            baseline: Optional personal baseline data.
            prodromal_panel: Optional preclinical Parkinson's panel.
            feedback_context: Optional additional context.
        """
        items = []
        priority = 1

        # 1. Prodromal-based recommendations (Highest priority if abnormal)
        if prodromal_panel and prodromal_panel.abnormal_count > 0:
            prod_items = self._prodromal_to_feedback(prodromal_panel, priority)
            items.extend(prod_items)
            priority += len(prod_items)

        # 2. Injury-based recommendations
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

        # 3. Deviation-based recommendations
        if deviation_report and deviation_report.alerts:
            for alert in deviation_report.alerts:
                item = self._deviation_to_feedback(alert, priority)
                if item:
                    items.append(item)
                    priority += 1

        # 4. General gait improvement tips based on baseline
        if baseline and baseline.num_sessions >= 1:
            general = self._baseline_tips(baseline, priority)
            items.extend(general)
            priority += len(general)

        # 5. Additional feedback loop items
        if feedback_context:
            for item in self._feedback_loop_items(feedback_context, priority):
                items.append(item)
                priority += 1

        # Overall status
        if injury_report.overall_risk >= 0.75 or (prodromal_panel and prodromal_panel.composite_score >= 0.3):
            status = "주의 필요"
            encouragement = "정밀 보행 분석 결과 주의가 필요한 소견이 있습니다. 아래 권장사항을 확인해주세요."
        elif injury_report.overall_risk >= 0.5 or (prodromal_panel and prodromal_panel.abnormal_count > 0):
            status = "개선 권장"
            encouragement = "몇 가지 개선이 필요하지만, 꾸준한 관리로 충분히 좋아질 수 있습니다!"
        elif injury_report.overall_risk >= 0.25:
            status = "양호"
            encouragement = "전체적으로 좋은 상태입니다. 작은 습관 개선으로 더 좋아질 수 있어요."
        else:
            status = "매우 양호"
            encouragement = "훌륭한 보행 패턴을 유지하고 있습니다! 계속 이대로 유지하세요."

        report = self._build_report(items, status, encouragement, injury_report, deviation_report, prodromal_panel)

        return PersonalizedFeedback(
            items=items,
            overall_status=status,
            encouragement=encouragement,
            report_kr=report,
        )

    def _prodromal_to_feedback(self, panel: ProdromalPanel, start_priority: int) -> list[FeedbackItem]:
        """Convert prodromal biomarkers to actionable feedback items."""
        items = []
        p = start_priority

        for b in panel.biomarkers:
            if not b.is_abnormal:
                continue

            if b.name in {"stride_time_cv", "cadence_variability"}:
                items.append(FeedbackItem(
                    category="exercise",
                    priority=p,
                    title="보행 리듬 및 규칙성 강화",
                    description=f"{b.korean_name}이 높게 측정되었습니다. 이는 보행 제어의 일관성이 저하되었음을 의미하며, 리듬 훈련이 도움이 됩니다.",
                    exercises=[
                        "메트로놈 보행 훈련 (분당 100-110회 박자에 맞춰 일정하게 걷기)",
                        "장애물 넘기 운동 (일정한 간격으로 놓인 선이나 장애물을 의식하며 걷기)",
                        "뒤로 걷기 (안전한 장소에서 천천히, 1분씩 3세트)",
                    ]
                ))
                p += 1
            elif b.name in {"rest_tremor_index", "nocturnal_movement_regularity"}:
                items.append(FeedbackItem(
                    category="medical",
                    priority=p,
                    title="신경계 전조 신호 관리",
                    description=f"{b.korean_name}에서 이상 소견이 보입니다. 이는 렘수면 행동 장애(RBD)와 연관된 미세 떨림이나 움직임 불규칙성일 수 있습니다.",
                    exercises=[
                        "수면 위생 개선 (암막 커튼 사용, 규칙적인 수면 시간)",
                        "취침 전 스트레칭 및 이완 요법 (10분)",
                        "신경과 전문의 상담 권장 (전임상 파킨슨 스크리닝 목적)",
                    ]
                ))
                p += 1
            elif b.name == "olfactory_screen_score":
                items.append(FeedbackItem(
                    category="medical",
                    priority=p,
                    title="후각 기능 자극 및 추적",
                    description="후각 기능 저하는 신경계 퇴행의 아주 이른 전조 증상 중 하나입니다.",
                    exercises=[
                        "후각 자극 훈련 (레몬, 장미, 유칼립투스 등 강한 향을 하루 2회 20초씩 맡기)",
                        "주기적인 보행 모니터링 및 인지 기능 체크",
                    ]
                ))
                p += 1

        return items

    def _feedback_loop_items(self, context: dict, start_priority: int) -> list[FeedbackItem]:
        """Map analysis outputs to correction actions and follow-up metrics.

        Expected context keys are intentionally simple so API/reporting layers can
        pass either raw biomarker dictionaries or summarized model outputs:
            biomarkers: dict[str, float]
            prodromal_stage: str
            prodromal_score: float
            target_accuracy: float
        """
        biomarkers = context.get("biomarkers") or {}
        items: list[FeedbackItem] = []
        p = start_priority

        def add(category: str, title: str, description: str, exercises: list[str]) -> None:
            nonlocal p
            items.append(FeedbackItem(category, p, title, description, exercises))
            p += 1

        stride_cv = float(biomarkers.get("stride_time_cv", biomarkers.get("acc_variability", 0.0)) or 0.0)
        cadence_var = float(biomarkers.get("cadence_variability", biomarkers.get("gyro_variability", 0.0)) or 0.0)
        asymmetry = float(biomarkers.get("pressure_asymmetry", biomarkers.get("right_left_acc_asymmetry", 0.0)) or 0.0)
        double_support = float(biomarkers.get("double_support_ratio", 0.0) or 0.0)
        prodromal_score = float(context.get("prodromal_score", 0.0) or 0.0)
        prodromal_stage = str(context.get("prodromal_stage", ""))

        if stride_cv >= 0.08 or cadence_var >= 0.10:
            add(
                "posture",
                "보행 리듬 안정화",
                "보폭 시간 또는 발목 회전 변동성이 높습니다. 다음 측정에서는 stride_time_cv와 cadence_variability 감소를 확인합니다.",
                [
                    "메트로놈 90-110 BPM 보행 10분",
                    "바닥 표식 간격에 맞춰 일정 보폭 걷기 5분",
                    "피로 시 세션을 중단하고 회복 후 재측정",
                ],
            )

        if asymmetry >= 0.12:
            add(
                "posture",
                "좌우 하중 대칭 교정",
                "좌우 발목 움직임 또는 족저압 비대칭이 큽니다. 다음 측정에서는 pressure_asymmetry와 right_left_acc_asymmetry를 추적합니다.",
                [
                    "거울 앞 체중 이동 좌우 10회씩 3세트",
                    "약한 쪽 다리 스텝업 10회씩 3세트",
                    "양발 지면 접촉 시간을 의식하며 천천히 걷기",
                ],
            )

        if double_support >= 0.32:
            add(
                "exercise",
                "이중 지지 시간 감소 훈련",
                "양발이 동시에 지면에 머무는 시간이 길어 안정성 보상 패턴이 의심됩니다. 다음 측정에서는 double_support_ratio를 확인합니다.",
                [
                    "보폭을 무리하게 늘리지 않고 발 구름을 부드럽게 연결",
                    "평지에서 5분 단위로 짧게 반복 보행",
                    "균형 불안이 지속되면 보호자 또는 전문가와 함께 훈련",
                ],
            )

        if prodromal_score >= 0.30 or prodromal_stage in {"prodromal", "전구기", "발현"}:
            add(
                "medical",
                "전조 신호 추적 강화",
                "전조 위험 점수가 높습니다. 보행 교정보다는 반복 측정과 전문 평가 연계를 우선합니다.",
                [
                    "동일 시간대, 동일 신발 조건으로 주 2회 이상 재측정",
                    "후각 선별, 안정 시 떨림, 보행 변동성 지표를 함께 기록",
                    "점수가 2주 이상 상승하면 신경과 상담 권장",
                ],
            )

        return items

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
        prodromal_panel: ProdromalPanel | None = None,
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

        # Prodromal summary
        if prodromal_panel:
            lines.append("─" * 60)
            lines.append("  [전임상 신경계 신호 분석]")
            lines.append("")
            lines.append(f"  위험 단계: {prodromal_panel.prodrome_stage} (위험 지수: {prodromal_panel.composite_score:.2f})")
            abnormal_biomarkers = [b for b in prodromal_panel.biomarkers if b.is_abnormal]
            if abnormal_biomarkers:
                lines.append("  감지된 이상 징후:")
                for b in abnormal_biomarkers:
                    lines.append(f"    ⚠ {b.korean_name}: {b.value:.4f} {b.unit} (정상: {b.normal_range[0]}~{b.normal_range[1]})")
            else:
                lines.append("  특이 사항 없음: 모든 전임상 바이오마커가 정상 범위 내에 있습니다.")
            lines.append("")

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
