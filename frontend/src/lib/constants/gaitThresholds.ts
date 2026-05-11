/** 룰 기반 보행·위험도 분석 임계값 (튜닝 가능, 하드코딩 분산 방지) */

export const GAIT_THRESHOLDS = {
  /** IMU 피크 검출: 인접 프레임 대비 최소 상대 높이 */
  peakProminence: 0.35,
  /** 최소 피크 간격 (프레임), 대략 100 steps/min 상한 고려 */
  minPeakDistanceFrames: 8,
  /** 정상 보행으로 보는 보조 대칭성 하한 */
  symmetryGood: 0.82,
  /** 리듬 변동성 경고 */
  rhythmCvWarn: 0.22,
  /** 동적 안정성(CoP 흔들림 프록시) 경고 */
  swayWarn: 0.07,
} as const;

export const RISK_WEIGHTS = {
  symmetry: 0.22,
  sway: 0.2,
  rhythm: 0.18,
  strideRegularity: 0.15,
  speed: 0.15,
  mlIndex: 0.1,
} as const;
