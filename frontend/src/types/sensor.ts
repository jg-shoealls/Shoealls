export type FootSide = "left" | "right";

export interface ImuSample {
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
}

export type PressureGrid = number[][];

export interface FootSensorFrame {
  timestampMs: number;
  side: FootSide;
  imu: ImuSample;
  pressure: PressureGrid;
}

export interface BilateralSensorFrame {
  timestampMs: number;
  left: FootSensorFrame;
  right: FootSensorFrame;
}

export interface SensorSeriesMeta {
  sampleRateHz: number;
  startedAtMs: number;
  frameCount: number;
}

export interface MultimodalSensorPayload {
  imu: number[][];
  pressure: PressureGrid;
  skeleton: number[][][];
}

export interface GaitFeatures {
  gait_speed: number;
  cadence: number;
  stride_regularity: number;
  step_symmetry: number;
  cop_sway: number;
  ml_index: number;
  arch_index: number;
  acceleration_rms: number;
  zone_heel_medial_mean: number;
  zone_heel_lateral_mean: number;
  zone_forefoot_medial_mean: number;
  zone_forefoot_lateral_mean: number;
  zone_toes_mean: number;
  zone_midfoot_medial_mean: number;
  zone_midfoot_lateral_mean: number;
  ml_variability?: number;
  heel_pressure_ratio?: number;
  forefoot_pressure_ratio?: number;
  pressure_asymmetry?: number;
  trunk_sway?: number;
}

export interface GaitEvent {
  type: "heel_strike" | "toe_off" | "stance_phase" | "swing_phase";
  side: FootSide;
  timestampMs: number;
  confidence: number;
}

export type RiskLevel = "low" | "watch" | "elevated" | "high";

export interface FallRiskResult {
  score: number;
  level: RiskLevel;
  label: string;
  reasons: string[];
  recommendation: string;
}

export interface GaitReport {
  score: number;
  symmetryPct: number;
  rhythmPct: number;
  stabilityPct: number;
  weightTransferPct: number;
  events: GaitEvent[];
  fallRisk: FallRiskResult;
  abnormalPatterns: string[];
}

export interface MockSensorData {
  meta: SensorSeriesMeta;
  frames: BilateralSensorFrame[];
  payload: MultimodalSensorPayload;
  features: GaitFeatures;
}

export interface InferenceInput {
  sensorData: MultimodalSensorPayload;
  features: GaitFeatures;
}

export interface InferenceOutput {
  report: GaitReport;
  modelName: string;
  isMock: boolean;
}

export interface GaitInferenceEngine {
  infer(input: InferenceInput): Promise<InferenceOutput>;
}
