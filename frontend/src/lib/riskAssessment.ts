/**
 * 룰 기반 낙상·보행 위험 지수 (0~100)
 * 질병 확정이 아닌 위험도·권고 수준 표현에만 사용합니다.
 */

import type { GaitFeatures } from "@/types/sensor";
import { RISK_WEIGHTS } from "./constants/gaitThresholds";

/** 룰 기반 위험도에 쓰는 최소 특성 (전체 GaitFeatures와 호환) */
export type GaitFeatureCore = Pick<
  GaitFeatures,
  "gait_speed" | "stride_regularity" | "step_symmetry" | "cop_sway" | "ml_index"
>;

export type RiskBand = "low" | "moderate" | "high";

export interface FallRiskAssessment {
  index: number;
  band: RiskBand;
  headline: string;
  detail: string;
  suggestedActions: string[];
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

/** 특성 기반 낙상 위험 지수 (높을수록 위험) */
export function computeFallRiskIndex(f: GaitFeatureCore): number {
  const symmetryRisk = clamp01((0.85 - f.step_symmetry) / 0.35);
  const swayRisk = clamp01((f.cop_sway - 0.04) / 0.12);
  const rhythmRisk = clamp01((0.65 - f.stride_regularity) / 0.5);
  const speedRisk = clamp01((1.0 - f.gait_speed) / 0.9);
  const mlRisk = clamp01(f.ml_index / 0.35);

  const raw =
    symmetryRisk * RISK_WEIGHTS.symmetry +
    swayRisk * RISK_WEIGHTS.sway +
    rhythmRisk * RISK_WEIGHTS.rhythm +
    speedRisk * RISK_WEIGHTS.speed +
    mlRisk * RISK_WEIGHTS.mlIndex;

  return Math.round(raw * 100);
}

function bandFromIndex(index: number): RiskBand {
  if (index < 35) return "low";
  if (index < 65) return "moderate";
  return "high";
}

export function assessFallRisk(f: GaitFeatureCore): FallRiskAssessment {
  const index = computeFallRiskIndex(f);
  const band = bandFromIndex(index);

  const headline =
    band === "low"
      ? "낙상 위험도: 관찰 수준"
      : band === "moderate"
        ? "낙상 위험도: 주의"
        : "낙상 위험도: 적극 관리 권고";

  const detail =
    band === "low"
      ? "현재 보행 특성은 상대적으로 안정적인 편입니다. 생활 습관 유지와 정기 모니터링을 권장합니다."
      : band === "moderate"
        ? "보행 안정성 관련 지표가 다소 불안정합니다. 보조 도구·환경 점검 및 전문 의료기관 방문을 검토해 보세요."
        : "동적 안정성·대칭성·속도 관련 위험 신호가 함께 나타납니다. 보호자 모니터링 강화 및 신속한 전문 평가를 권고합니다.";

  const suggestedActions: string[] = [];
  if (band !== "low") {
    suggestedActions.push("실내 장애물·미끄러운 바닥 점검");
    suggestedActions.push("보행 보조기구 필요 여부 상담");
  }
  if (band === "high") {
    suggestedActions.push("가까운 전문 의료기관 보행·근력 평가 예약 검토");
  }
  if (suggestedActions.length === 0) {
    suggestedActions.push("주간 단위로 동일 조건에서 재측정해 추세를 확인하세요.");
  }

  return { index, band, headline, detail, suggestedActions };
}

/** 보행 점수 0~100 (높을수록 양호) — 리포트 카드용 */
export function computeGaitQualityScore(f: GaitFeatureCore): number {
  const fall = computeFallRiskIndex(f);
  const base = 100 - fall;
  const bonus =
    f.stride_regularity > 0.75 && f.step_symmetry > 0.82 ? 5 : 0;
  return Math.max(0, Math.min(100, base + bonus));
}
