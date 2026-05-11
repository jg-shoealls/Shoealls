/**
 * 룰 기반 보행 분석 유틸리티
 * FeatureLike, GaitAnalysisSummary, analyzeGaitRuleBased (AI 모델 연동 전 룰 기반)
 * detectGaitEvents, calculateFallRisk, buildGaitReport (기존 함수 유지)
 */

import type {
  BilateralSensorFrame,
  FallRiskResult,
  FootSide,
  GaitEvent,
  GaitFeatures,
  GaitReport,
  MultimodalSensorPayload,
  PressureGrid,
} from "@/types/sensor";
import { GAIT_THRESHOLDS } from "./constants/gaitThresholds";

// ─── 공용 타입 ────────────────────────────────────────────────────────────────

/** 룰 기반 분석에 필요한 최소 특성 집합 (GaitFeatures 서브셋) */
export interface FeatureLike {
  gait_speed: number;
  stride_regularity: number;
  step_symmetry: number;
  cop_sway: number;
  ml_index: number;
}

/** analyzeGaitRuleBased 반환 타입 */
export interface GaitAnalysisSummary {
  rhythm: {
    strideTimeCv: number;
    estimatedCadenceStepsPerMin: number;
    note: string;
  };
  symmetry: {
    score01: number;
    label: string;
  };
  stability: {
    dynamicStability01: number;
    note: string;
  };
  anomalies: string[];
}

// ─── 내부 유틸 ────────────────────────────────────────────────────────────────

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

function pressureSum(grid: PressureGrid): number {
  return grid.flat().reduce((sum, v) => sum + v, 0);
}

function footPressure(frame: BilateralSensorFrame, side: FootSide): number {
  return pressureSum(frame[side].pressure);
}

/** 신호에서 로컬 피크 인덱스 배열 반환 (단순 prominence 기반) */
function findPeaks(signal: number[], minDist: number, prominenceRatio: number): number[] {
  if (signal.length < 3) return [];
  const maxVal = Math.max(...signal);
  const threshold = maxVal * prominenceRatio;
  const peaks: number[] = [];

  for (let i = 1; i < signal.length - 1; i++) {
    if (
      signal[i] > signal[i - 1] &&
      signal[i] > signal[i + 1] &&
      signal[i] > threshold
    ) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDist) {
        peaks.push(i);
      }
    }
  }
  return peaks;
}

/** 변동계수(CV) 계산 */
function computeCv(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  if (mean === 0) return 0;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
  return parseFloat((Math.sqrt(variance) / mean).toFixed(3));
}

// ─── 핵심 룰 기반 분석 ────────────────────────────────────────────────────────

/**
 * 멀티모달 센서 데이터 + 특성값으로 보행 분석 요약 생성.
 * 향후 LSTM / Transformer 추론 결과로 교체 가능한 인터페이스.
 */
export function analyzeGaitRuleBased(
  payload: MultimodalSensorPayload,
  features: FeatureLike,
  sampleRateHz: number
): GaitAnalysisSummary {
  // 수직 가속도(az, 채널 2) 절댓값으로 보폭 피크 검출
  const az = payload.imu.map((frame) => Math.abs(frame[2] ?? 0));
  const peaks = findPeaks(az, GAIT_THRESHOLDS.minPeakDistanceFrames, GAIT_THRESHOLDS.peakProminence);

  const intervalFrames = peaks.slice(1).map((p, i) => p - peaks[i]);
  const intervalsSec = intervalFrames.map((f) => f / sampleRateHz);
  const strideTimeCv = computeCv(intervalsSec);

  let estimatedCadenceStepsPerMin: number;
  if (intervalsSec.length >= 2) {
    const mean = intervalsSec.reduce((a, b) => a + b, 0) / intervalsSec.length;
    estimatedCadenceStepsPerMin = Math.round(60 / mean);
  } else {
    // 피크 검출 실패 시 속도 기반 추정
    estimatedCadenceStepsPerMin = Math.round(features.gait_speed * 88);
  }
  estimatedCadenceStepsPerMin = Math.max(40, Math.min(160, estimatedCadenceStepsPerMin));

  const rhythmNote =
    strideTimeCv < GAIT_THRESHOLDS.rhythmCvWarn
      ? `보폭 리듬 변동성 CV=${strideTimeCv.toFixed(3)} — 안정적 범주`
      : `보폭 리듬 변동성 CV=${strideTimeCv.toFixed(3)} — 불규칙 징후 감지`;

  // 좌우 대칭성
  const symmetryScore01 = clamp01(features.step_symmetry);
  const symmetryLabel =
    symmetryScore01 >= GAIT_THRESHOLDS.symmetryGood
      ? `좌우 대칭성 ${(symmetryScore01 * 100).toFixed(1)}% — 정상 범주`
      : `좌우 대칭성 ${(symmetryScore01 * 100).toFixed(1)}% — 비대칭 징후`;

  // 동적 안정성 (CoP 흔들림 역산)
  const dynamicStability01 = clamp01(1 - (features.cop_sway - 0.02) / 0.12);
  const stabilityNote =
    features.cop_sway > GAIT_THRESHOLDS.swayWarn
      ? `CoP 흔들림 ${features.cop_sway.toFixed(3)} — 동적 안정성 저하 징후`
      : `CoP 흔들림 ${features.cop_sway.toFixed(3)} — 안정적 범주`;

  // 이상 패턴 감지 (비진단, 위험 징후 수준)
  const anomalies: string[] = [];
  if (strideTimeCv > GAIT_THRESHOLDS.rhythmCvWarn) anomalies.push("보행 리듬 불규칙 징후");
  if (symmetryScore01 < GAIT_THRESHOLDS.symmetryGood) anomalies.push("좌우 보행 비대칭 징후");
  if (features.cop_sway > GAIT_THRESHOLDS.swayWarn) anomalies.push("동적 안정성 저하 징후");
  if (features.gait_speed < 0.8) anomalies.push("보행 속도 저하 징후");
  if (features.ml_index > 0.15) anomalies.push("내외측 체중 이동 이상 징후");

  return {
    rhythm: { strideTimeCv, estimatedCadenceStepsPerMin, note: rhythmNote },
    symmetry: { score01: symmetryScore01, label: symmetryLabel },
    stability: { dynamicStability01, note: stabilityNote },
    anomalies,
  };
}

// ─── 기존 함수 (하위 호환 유지) ───────────────────────────────────────────────

export function detectGaitEvents(frames: BilateralSensorFrame[]): GaitEvent[] {
  const events: GaitEvent[] = [];
  const previous: Record<FootSide, number> = { left: 0, right: 0 };
  const active: Record<FootSide, boolean> = { left: false, right: false };

  for (const frame of frames) {
    for (const side of ["left", "right"] as const) {
      const current = footPressure(frame, side);
      const delta = current - previous[side];
      const isLoaded = current > 18;

      if (!active[side] && isLoaded && delta > 2) {
        events.push({
          type: "heel_strike",
          side,
          timestampMs: frame.timestampMs,
          confidence: clamp01(delta / 10),
        });
        active[side] = true;
      }

      if (active[side] && current < 14 && delta < -2) {
        events.push({
          type: "toe_off",
          side,
          timestampMs: frame.timestampMs,
          confidence: clamp01(Math.abs(delta) / 10),
        });
        active[side] = false;
      }

      previous[side] = current;
    }
  }
  return events;
}

export function calculateFallRisk(features: GaitFeatures): FallRiskResult {
  const speedRisk = clamp01((0.95 - features.gait_speed) / 0.55);
  const symmetryRisk = clamp01((0.86 - features.step_symmetry) / 0.36);
  const rhythmRisk = clamp01((0.75 - features.stride_regularity) / 0.4);
  const swayRisk = clamp01((features.cop_sway - 0.04) / 0.08);
  const mlRisk = clamp01((features.ml_index - 0.05) / 0.2);
  const accelerationRisk = clamp01((1.25 - (features.acceleration_rms ?? 1.0)) / 0.75);

  const score = clamp01(
    speedRisk * 0.2 +
      symmetryRisk * 0.18 +
      rhythmRisk * 0.2 +
      swayRisk * 0.22 +
      mlRisk * 0.12 +
      accelerationRisk * 0.08,
  );

  const reasons: string[] = [];
  if (speedRisk > 0.45) reasons.push("보행 속도 저하");
  if (symmetryRisk > 0.45) reasons.push("좌우 대칭성 저하");
  if (rhythmRisk > 0.45) reasons.push("보행 리듬 불규칙");
  if (swayRisk > 0.45) reasons.push("체중 중심 흔들림 증가");
  if (mlRisk > 0.45) reasons.push("좌우 방향 변동성 증가");

  if (score >= 0.7) return { score, level: "high", label: "높음", reasons, recommendation: "최근 보행 변화가 반복되면 보호자와 공유하고 전문 의료기관 상담을 권고합니다." };
  if (score >= 0.5) return { score, level: "elevated", label: "주의", reasons, recommendation: "보행 환경을 점검하고 일주일 추세를 추가 관찰하세요." };
  if (score >= 0.3) return { score, level: "watch", label: "관찰", reasons, recommendation: "피로, 신발 상태, 활동량 변화와 함께 모니터링하세요." };
  return { score, level: "low", label: "낮음", reasons: reasons.length > 0 ? reasons : ["주요 위험 지표 안정"], recommendation: "현재 기준에서는 급격한 위험 신호가 낮습니다." };
}

export function buildGaitReport(features: GaitFeatures, frames: BilateralSensorFrame[]): GaitReport {
  const fallRisk = calculateFallRisk(features);
  const events = detectGaitEvents(frames);
  const symmetryPct = Math.round(clamp01(features.step_symmetry) * 100);
  const rhythmPct = Math.round(clamp01(features.stride_regularity) * 100);
  const stabilityPct = Math.round((1 - clamp01((features.cop_sway - 0.02) / 0.1)) * 100);
  const weightTransferPct = Math.round((1 - clamp01(features.ml_index / 0.25)) * 100);
  const score = Math.round(
    symmetryPct * 0.25 + rhythmPct * 0.25 + stabilityPct * 0.3 + weightTransferPct * 0.2,
  );

  const abnormalPatterns = [
    features.step_symmetry < 0.82 ? "좌우 보행 비대칭 징후" : null,
    features.stride_regularity < 0.65 ? "보행 리듬 저하 징후" : null,
    features.cop_sway > 0.07 ? "동적 안정성 저하 징후" : null,
    features.gait_speed < 0.8 ? "보행 속도 저하 징후" : null,
  ].filter((item): item is string => Boolean(item));

  return { score, symmetryPct, rhythmPct, stabilityPct, weightTransferPct, events, fallRisk, abnormalPatterns };
}
