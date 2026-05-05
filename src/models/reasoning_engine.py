"""추론 엔진: 단계별 근거 기반 보행 분석 AI.

Chain-of-Reasoning 아키텍처:
    1단계: 모달리티별 이상 패턴 감지 (Anomaly Detection)
    2단계: 교차 모달 근거 수집 (Cross-Modal Evidence)
    3단계: 감별 진단 추론 (Differential Diagnosis)
    4단계: 신뢰도 보정 및 최종 판정 (Calibrated Decision)

각 단계가 다음 단계의 입력이 되는 자기회귀적 추론 체인.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDetectionModule(nn.Module):
    """1단계: 모달리티별 이상 패턴 감지.

    각 센서 데이터에서 정상 보행 대비 편차를 감지하고,
    어느 시간대/어느 센서 영역이 비정상인지 히트맵을 생성.
    """

    def __init__(self, embed_dim: int = 128, num_anomaly_types: int = 8):
        super().__init__()

        # 학습된 정상 보행 프로토타입
        self.normal_prototype = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # 이상 유형별 검출기
        # 0: 비대칭, 1: 리듬 불규칙, 2: 진폭 이상, 3: 주파수 이상
        # 4: 공간 패턴 이상, 5: 시간 지연, 6: 떨림, 7: 동결
        self.anomaly_detectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
            for _ in range(num_anomaly_types)
        ])

        # 시간축 이상 히트맵 생성기
        self.temporal_anomaly_conv = nn.Conv1d(embed_dim, 1, kernel_size=5, padding=2)

        self.num_anomaly_types = num_anomaly_types

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: (B, T, D) 인코더 출력.

        Returns:
            anomaly_scores: (B, num_anomaly_types) 이상 유형별 점수
            temporal_heatmap: (B, T) 시간축 이상 히트맵
            deviation: (B, D) 정상 대비 편차 벡터
        """
        # 시간 평균 특징
        pooled = features.mean(dim=1)  # (B, D)

        # 정상 프로토타입 대비 편차
        deviation = pooled - self.normal_prototype  # (B, D)

        # 이상 유형별 점수
        anomaly_scores = torch.stack([
            det(deviation).squeeze(-1) for det in self.anomaly_detectors
        ], dim=1)  # (B, num_anomaly_types)
        anomaly_scores = torch.sigmoid(anomaly_scores)

        # 시간축 이상 히트맵
        temporal_heatmap = self.temporal_anomaly_conv(
            features.permute(0, 2, 1)
        ).squeeze(1)  # (B, T)
        temporal_heatmap = torch.sigmoid(temporal_heatmap)

        return {
            "anomaly_scores": anomaly_scores,
            "temporal_heatmap": temporal_heatmap,
            "deviation": deviation,
        }


class CrossModalEvidenceCollector(nn.Module):
    """2단계: 교차 모달 근거 수집.

    각 모달리티의 이상 패턴이 다른 모달리티에서도 확인되는지
    교차 검증하여 근거 강도를 계산.

    예: IMU에서 비대칭 감지 → 족저압에서도 좌우 불균형 확인 → 근거 강화
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()

        # 모달리티 간 교차 검증 어텐션
        self.cross_verify = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 근거 강도 계산기
        self.evidence_strength = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # 모달리티별 기여도 게이트
        self.modality_gate = nn.Sequential(
            nn.Linear(embed_dim, 3),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        modality_features: list[torch.Tensor],
        anomaly_deviations: list[torch.Tensor],
    ) -> dict:
        """
        Args:
            modality_features: [imu_feat, pressure_feat, skeleton_feat] 각 (B, T_i, D)
            anomaly_deviations: [imu_dev, pressure_dev, skeleton_dev] 각 (B, D)

        Returns:
            evidence_embedding: (B, D) 근거 융합 임베딩
            modality_weights: (B, 3) 모달리티별 기여도
            cross_support: (B, 3) 교차 검증 지지도
        """
        B = modality_features[0].size(0)
        D = modality_features[0].size(2)

        # 모달리티별 요약
        summaries = torch.stack([f.mean(dim=1) for f in modality_features], dim=1)  # (B, 3, D)

        # 교차 검증: 각 모달리티가 다른 모달리티를 참조
        cross_out, _ = self.cross_verify(
            summaries, summaries, summaries, need_weights=False
        )
        cross_out = self.norm(summaries + cross_out)  # (B, 3, D)

        # 교차 지지도: 다른 모달리티와의 일치도
        deviations = torch.stack(anomaly_deviations, dim=1)  # (B, 3, D)
        cross_support = F.cosine_similarity(
            deviations.unsqueeze(2).expand(-1, -1, 3, -1),
            deviations.unsqueeze(1).expand(-1, 3, -1, -1),
            dim=-1,
        )  # (B, 3, 3)
        # 자기 자신을 제외한 평균 지지도
        mask = ~torch.eye(3, dtype=torch.bool, device=cross_support.device)  # (3, 3)
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, 3, 3)
        cross_support_masked = cross_support.masked_fill(~mask, 0.0)
        cross_support_score = cross_support_masked.sum(dim=-1) / 2.0  # (B, 3)

        # 모달리티 기여도 게이트
        gate_input = cross_out.mean(dim=1)  # (B, D)
        modality_weights = self.modality_gate(gate_input)  # (B, 3)

        # 가중 융합
        weighted = (cross_out * modality_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # 근거 강도
        combined = torch.cat([weighted, gate_input], dim=-1)
        strength = self.evidence_strength(combined)  # (B, 1)

        return {
            "evidence_embedding": weighted,
            "modality_weights": modality_weights,
            "cross_support": cross_support_score,
            "evidence_strength": strength.squeeze(-1),
        }


class DifferentialDiagnosisChain(nn.Module):
    """3단계: 감별 진단 추론 체인.

    질환 프로토타입과 비교하여 가능한 진단을 순위화하고,
    각 진단에 대한 찬성/반대 근거를 추론.

    Chain-of-Thought 구조:
        가설 생성 → 근거 대조 → 가설 업데이트 → 최종 순위
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 4,
        num_reasoning_steps: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_steps = num_reasoning_steps

        # 질환별 학습 프로토타입 (각 질환의 전형적 특징)
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, embed_dim) * 0.02
        )

        # 추론 단계별 Transformer 블록
        self.reasoning_steps = nn.ModuleList([
            ReasoningBlock(embed_dim, num_classes)
            for _ in range(num_reasoning_steps)
        ])

        # 찬성/반대 근거 생성기
        self.pro_evidence = nn.Linear(embed_dim, num_classes)
        self.con_evidence = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        evidence_embedding: torch.Tensor,
        anomaly_context: torch.Tensor,
    ) -> dict:
        """
        Args:
            evidence_embedding: (B, D) 2단계 출력.
            anomaly_context: (B, D) 이상 패턴 정보.

        Returns:
            hypothesis_logits: (B, num_classes) 진단 점수
            reasoning_trace: list of (B, num_classes) 각 추론 단계 점수
            pro_scores: (B, num_classes) 찬성 근거 강도
            con_scores: (B, num_classes) 반대 근거 강도
        """
        B = evidence_embedding.size(0)

        # 초기 가설: 프로토타입과의 유사도
        similarity = F.cosine_similarity(
            evidence_embedding.unsqueeze(1),          # (B, 1, D)
            self.class_prototypes.unsqueeze(0),        # (1, C, D)
            dim=-1,
        )  # (B, C)

        # 단계별 추론
        hypothesis = similarity
        context = evidence_embedding + anomaly_context  # (B, D)
        reasoning_trace = [hypothesis.clone()]

        for step in self.reasoning_steps:
            hypothesis, context = step(hypothesis, context, self.class_prototypes)
            reasoning_trace.append(hypothesis.clone())

        # 찬성/반대 근거
        pro = torch.sigmoid(self.pro_evidence(context))   # (B, C)
        con = torch.sigmoid(self.con_evidence(context))   # (B, C)

        return {
            "hypothesis_logits": hypothesis,
            "reasoning_trace": reasoning_trace,
            "pro_scores": pro,
            "con_scores": con,
        }


class ReasoningBlock(nn.Module):
    """단일 추론 단계: 가설 → 근거 대조 → 업데이트."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()

        # 가설과 근거를 결합하여 업데이트
        self.hypothesis_update = nn.Sequential(
            nn.Linear(embed_dim + num_classes, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

        # 컨텍스트 정제
        self.context_refine = nn.Sequential(
            nn.Linear(embed_dim + num_classes, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        hypothesis: torch.Tensor,  # (B, C)
        context: torch.Tensor,      # (B, D)
        prototypes: torch.Tensor,    # (C, D)
    ) -> tuple:
        # 현재 가설 기반으로 프로토타입 가중 합산
        attn = F.softmax(hypothesis, dim=-1)                  # (B, C)
        attended_proto = torch.mm(attn, prototypes)            # (B, D)

        # 컨텍스트와 프로토타입의 차이로 가설 업데이트
        combined = torch.cat([context, hypothesis], dim=-1)    # (B, D+C)
        delta = self.hypothesis_update(combined)               # (B, C)
        new_hypothesis = hypothesis + delta                    # residual

        # 컨텍스트 정제
        ctx_combined = torch.cat([context + attended_proto, hypothesis], dim=-1)
        new_context = self.context_refine(ctx_combined)

        return new_hypothesis, new_context


class ConfidenceCalibrator(nn.Module):
    """4단계: 신뢰도 보정 및 최종 판정.

    추론 체인의 일관성, 근거 강도, 교차 검증 지지도를
    종합하여 보정된 최종 확률을 출력.
    """

    def __init__(self, num_classes: int = 4, num_reasoning_steps: int = 3):
        super().__init__()

        # 입력: 최종 가설 + 근거 강도 + 찬반 + 추론 일관성
        input_dim = num_classes * 3 + 1 + num_reasoning_steps + 1

        self.calibration_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

        # Temperature scaling (학습 가능한 보정)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(
        self,
        hypothesis_logits: torch.Tensor,
        pro_scores: torch.Tensor,
        con_scores: torch.Tensor,
        evidence_strength: torch.Tensor,
        reasoning_trace: list[torch.Tensor],
        cross_support_mean: torch.Tensor,
    ) -> dict:
        """
        Returns:
            calibrated_logits: (B, C) 보정된 최종 로짓
            calibrated_probs: (B, C) 보정된 확률
            uncertainty: (B,) 예측 불확실성
        """
        # 추론 일관성: 각 단계 간 변화량
        consistency = []
        for i in range(1, len(reasoning_trace)):
            delta = (reasoning_trace[i] - reasoning_trace[i-1]).abs().mean(dim=-1)
            consistency.append(delta)
        consistency = torch.stack(consistency, dim=1)  # (B, num_steps)

        # 입력 결합
        features = torch.cat([
            hypothesis_logits,      # (B, C)
            pro_scores,             # (B, C)
            con_scores,             # (B, C)
            evidence_strength.unsqueeze(-1),  # (B, 1)
            consistency,            # (B, num_steps)
            cross_support_mean.unsqueeze(-1),  # (B, 1)
        ], dim=-1)

        calibrated_logits = self.calibration_net(features)

        # Temperature scaling
        calibrated_probs = F.softmax(calibrated_logits / self.temperature, dim=-1)

        # 불확실성: 엔트로피 기반
        entropy = -(calibrated_probs * (calibrated_probs + 1e-8).log()).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(calibrated_probs.size(-1))))
        uncertainty = entropy / max_entropy  # 0~1 정규화

        return {
            "calibrated_logits": calibrated_logits,
            "calibrated_probs": calibrated_probs,
            "uncertainty": uncertainty,
        }


class GaitReasoningEngine(nn.Module):
    """멀티모달 보행 분석 추론 엔진.

    4단계 Chain-of-Reasoning:
        Step 1: 이상 감지 — 각 센서에서 비정상 패턴 탐지
        Step 2: 근거 수집 — 교차 모달 검증으로 근거 강화
        Step 3: 감별 진단 — 가설 생성→대조→업데이트 반복
        Step 4: 신뢰도 보정 — 추론 일관성 기반 최종 판정

    사용법:
        engine = GaitReasoningEngine(config)
        result = engine.reason(batch)    # 추론 실행
        report = engine.explain(result)  # 한글 추론 리포트 생성
    """

    ANOMALY_NAMES_KR = [
        "좌우 비대칭", "리듬 불규칙", "진폭 이상", "주파수 이상",
        "공간 패턴 이상", "시간 지연", "떨림", "보행 동결",
    ]

    CLASS_NAMES_KR = ["정상 보행", "절뚝거림", "운동실조", "파킨슨"]

    MODALITY_NAMES_KR = ["IMU (관성센서)", "족저압 센서", "스켈레톤"]

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        data_cfg = config["data"]
        embed_dim = model_cfg["fusion"]["embed_dim"]
        num_classes = data_cfg["num_classes"]

        reasoning_cfg = config.get("reasoning", {})
        num_reasoning_steps = reasoning_cfg.get("num_steps", 3)

        # 공유 인코더 (기존 모델에서 로드)
        from .encoders import IMUEncoder, PressureEncoder, SkeletonEncoder
        imu_cfg = model_cfg["imu_encoder"]
        self.imu_encoder = IMUEncoder(
            in_channels=data_cfg["imu_channels"],
            conv_channels=imu_cfg["conv_channels"],
            kernel_size=imu_cfg["kernel_size"],
            lstm_hidden=embed_dim,
            lstm_layers=imu_cfg["lstm_layers"],
            dropout=imu_cfg["dropout"],
        )
        pressure_cfg = model_cfg["pressure_encoder"]
        self.pressure_encoder = PressureEncoder(
            in_channels=1,
            conv_channels=pressure_cfg["conv_channels"],
            kernel_size=pressure_cfg["kernel_size"],
            embed_dim=embed_dim,
            dropout=pressure_cfg["dropout"],
        )
        skeleton_cfg = model_cfg["skeleton_encoder"]
        self.skeleton_encoder = SkeletonEncoder(
            in_channels=data_cfg["skeleton_dims"],
            num_joints=data_cfg["skeleton_joints"],
            gcn_channels=skeleton_cfg["gcn_channels"],
            temporal_kernel=skeleton_cfg["temporal_kernel"],
            embed_dim=embed_dim,
            dropout=skeleton_cfg["dropout"],
        )

        # 4단계 추론 모듈
        self.anomaly_detectors = nn.ModuleList([
            AnomalyDetectionModule(embed_dim) for _ in range(3)
        ])
        self.evidence_collector = CrossModalEvidenceCollector(embed_dim)
        self.diagnosis_chain = DifferentialDiagnosisChain(
            embed_dim, num_classes, num_reasoning_steps,
        )
        self.confidence_calibrator = ConfidenceCalibrator(
            num_classes, num_reasoning_steps,
        )

    def forward(self, batch: dict) -> dict:
        """기본 forward (학습용)."""
        result = self.reason(batch)
        return result["calibrated_logits"]

    @torch.no_grad()
    def reason(self, batch: dict) -> dict:
        """전체 추론 체인 실행."""
        self.eval()

        # 인코딩
        imu_feat = self.imu_encoder(batch["imu"])
        pressure_feat = self.pressure_encoder(batch["pressure"])
        skeleton_feat = self.skeleton_encoder(batch["skeleton"])
        modality_features = [imu_feat, pressure_feat, skeleton_feat]

        # Step 1: 이상 감지
        anomaly_results = []
        for feat, detector in zip(modality_features, self.anomaly_detectors):
            anomaly_results.append(detector(feat))

        # Step 2: 근거 수집
        deviations = [r["deviation"] for r in anomaly_results]
        evidence = self.evidence_collector(modality_features, deviations)

        # Step 3: 감별 진단
        anomaly_context = sum(deviations) / 3
        diagnosis = self.diagnosis_chain(
            evidence["evidence_embedding"], anomaly_context
        )

        # Step 4: 신뢰도 보정
        calibration = self.confidence_calibrator(
            hypothesis_logits=diagnosis["hypothesis_logits"],
            pro_scores=diagnosis["pro_scores"],
            con_scores=diagnosis["con_scores"],
            evidence_strength=evidence["evidence_strength"],
            reasoning_trace=diagnosis["reasoning_trace"],
            cross_support_mean=evidence["cross_support"].mean(dim=-1),
        )

        return {
            # 최종 결과
            "calibrated_logits": calibration["calibrated_logits"],
            "calibrated_probs": calibration["calibrated_probs"],
            "uncertainty": calibration["uncertainty"],
            "prediction": calibration["calibrated_probs"].argmax(dim=-1),
            # 추론 과정 (설명용)
            "anomaly_results": anomaly_results,
            "evidence": evidence,
            "diagnosis": diagnosis,
        }

    def explain(self, result: dict, sample_idx: int = 0) -> str:
        """추론 결과를 한글 리포트로 변환.

        Args:
            result: reason() 출력.
            sample_idx: 배치 내 샘플 인덱스.

        Returns:
            한글 추론 리포트 문자열.
        """
        i = sample_idx
        pred = result["prediction"][i].item()
        probs = result["calibrated_probs"][i].cpu().numpy()
        uncertainty = result["uncertainty"][i].item()

        lines = []
        lines.append("=" * 60)
        lines.append("  멀티모달 보행 분석 AI 추론 리포트")
        lines.append("=" * 60)

        # ── 최종 판정 ──
        lines.append("")
        lines.append(f"  최종 판정: {self.CLASS_NAMES_KR[pred]}")
        lines.append(f"  확신도:    {probs[pred]:.1%}")
        lines.append(f"  불확실성:  {uncertainty:.1%}")
        lines.append("")

        # ── 1단계: 이상 패턴 감지 ──
        lines.append("-" * 60)
        lines.append("  [1단계] 모달리티별 이상 패턴 감지")
        lines.append("-" * 60)

        for m_idx, (m_name, anom) in enumerate(
            zip(self.MODALITY_NAMES_KR, result["anomaly_results"])
        ):
            scores = anom["anomaly_scores"][i].cpu().numpy()
            top_anomalies = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )[:3]

            detected = [(idx, s) for idx, s in top_anomalies if s > 0.5]
            if detected:
                findings = ", ".join(
                    f"{self.ANOMALY_NAMES_KR[idx]}({s:.0%})"
                    for idx, s in detected
                )
                lines.append(f"  {m_name}: {findings}")
            else:
                lines.append(f"  {m_name}: 주요 이상 패턴 미감지")

        # ── 2단계: 교차 검증 ──
        lines.append("")
        lines.append("-" * 60)
        lines.append("  [2단계] 교차 모달 근거 검증")
        lines.append("-" * 60)

        weights = result["evidence"]["modality_weights"][i].cpu().numpy()
        cross_support = result["evidence"]["cross_support"][i].cpu().numpy()
        strength = result["evidence"]["evidence_strength"][i].item()

        for m_idx, m_name in enumerate(self.MODALITY_NAMES_KR):
            lines.append(
                f"  {m_name}: "
                f"기여도 {weights[m_idx]:.0%} | "
                f"교차지지도 {cross_support[m_idx]:.0%}"
            )
        lines.append(f"  종합 근거 강도: {strength:.0%}")

        # ── 3단계: 감별 진단 ──
        lines.append("")
        lines.append("-" * 60)
        lines.append("  [3단계] 감별 진단 추론 과정")
        lines.append("-" * 60)

        trace = result["diagnosis"]["reasoning_trace"]
        pro = result["diagnosis"]["pro_scores"][i].cpu().numpy()
        con = result["diagnosis"]["con_scores"][i].cpu().numpy()

        for step_idx, step_logits in enumerate(trace):
            step_probs = F.softmax(step_logits[i], dim=-1).cpu().numpy()
            top_cls = step_probs.argmax()
            label = "초기 가설" if step_idx == 0 else f"추론 {step_idx}단계"
            lines.append(
                f"  {label}: "
                f"{self.CLASS_NAMES_KR[top_cls]} ({step_probs[top_cls]:.0%})"
            )

        lines.append("")
        lines.append("  진단별 근거 분석:")
        ranked = sorted(range(len(probs)), key=lambda c: probs[c], reverse=True)
        for cls_idx in ranked:
            marker = ">>" if cls_idx == pred else "  "
            lines.append(
                f"  {marker} {self.CLASS_NAMES_KR[cls_idx]:10s} "
                f"확률 {probs[cls_idx]:5.1%} | "
                f"찬성 {pro[cls_idx]:.0%} | "
                f"반대 {con[cls_idx]:.0%}"
            )

        # ── 4단계: 신뢰도 판정 ──
        lines.append("")
        lines.append("-" * 60)
        lines.append("  [4단계] 최종 신뢰도 판정")
        lines.append("-" * 60)

        if uncertainty < 0.2:
            confidence_level = "높음 (확신할 수 있는 판정)"
        elif uncertainty < 0.5:
            confidence_level = "보통 (추가 검증 권장)"
        else:
            confidence_level = "낮음 (전문의 확인 필요)"

        lines.append(f"  판정 신뢰 수준: {confidence_level}")

        # 추론 일관성
        changes = []
        for s in range(1, len(trace)):
            prev_top = F.softmax(trace[s-1][i], dim=-1).argmax().item()
            curr_top = F.softmax(trace[s][i], dim=-1).argmax().item()
            if prev_top != curr_top:
                changes.append(
                    f"  단계{s}: {self.CLASS_NAMES_KR[prev_top]} → {self.CLASS_NAMES_KR[curr_top]}"
                )

        if changes:
            lines.append("  추론 과정 중 가설 변경:")
            lines.extend(changes)
        else:
            lines.append("  추론 전 단계에서 가설 일관성 유지")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def load_base_model_weights(self, checkpoint_path: str, device="cpu"):
        """기존 학습된 모델에서 인코더 가중치를 로드."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint["model_state_dict"]

        # 인코더 가중치만 추출하여 로드
        encoder_mapping = {
            "imu_encoder.": "imu_encoder.",
            "pressure_encoder.": "pressure_encoder.",
            "skeleton_encoder.": "skeleton_encoder.",
        }

        own_state = self.state_dict()
        loaded = 0
        for key, value in state.items():
            for prefix in encoder_mapping:
                if key.startswith(prefix):
                    target_key = encoder_mapping[prefix] + key[len(prefix):]
                    if target_key in own_state:
                        own_state[target_key].copy_(value)
                        loaded += 1

        print(f"Loaded {loaded} encoder parameters from base model")
