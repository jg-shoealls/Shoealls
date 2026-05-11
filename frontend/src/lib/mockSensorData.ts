import type {
  BilateralSensorFrame,
  FootSensorFrame,
  GaitFeatures,
  ImuSample,
  MockSensorData,
  PressureGrid,
} from "@/types/sensor";

export type MockProfile = "normal" | "parkinsons" | "stroke" | "fall_risk";

export const PROFILE_LABELS: Record<MockProfile, string> = {
  normal: "안정 보행",
  parkinsons: "짧은 보폭 패턴",
  stroke: "편측 비대칭 패턴",
  fall_risk: "낙상 위험 관찰 패턴",
};

const PROFILE_FEATURES: Record<MockProfile, GaitFeatures> = {
  normal: {
    gait_speed: 1.22,
    cadence: 112,
    stride_regularity: 0.88,
    step_symmetry: 0.93,
    cop_sway: 0.034,
    ml_index: 0.03,
    arch_index: 0.24,
    acceleration_rms: 1.55,
    zone_heel_medial_mean: 0.32,
    zone_heel_lateral_mean: 0.3,
    zone_forefoot_medial_mean: 0.34,
    zone_forefoot_lateral_mean: 0.31,
    zone_toes_mean: 0.18,
    zone_midfoot_medial_mean: 0.1,
    zone_midfoot_lateral_mean: 0.08,
  },
  parkinsons: {
    gait_speed: 0.68,
    cadence: 144,
    stride_regularity: 0.54,
    step_symmetry: 0.79,
    cop_sway: 0.064,
    ml_index: 0.09,
    arch_index: 0.23,
    acceleration_rms: 0.92,
    zone_heel_medial_mean: 0.28,
    zone_heel_lateral_mean: 0.25,
    zone_forefoot_medial_mean: 0.38,
    zone_forefoot_lateral_mean: 0.35,
    zone_toes_mean: 0.12,
    zone_midfoot_medial_mean: 0.1,
    zone_midfoot_lateral_mean: 0.1,
  },
  stroke: {
    gait_speed: 0.52,
    cadence: 84,
    stride_regularity: 0.52,
    step_symmetry: 0.58,
    cop_sway: 0.086,
    ml_index: 0.21,
    arch_index: 0.27,
    acceleration_rms: 1.0,
    zone_heel_medial_mean: 0.18,
    zone_heel_lateral_mean: 0.35,
    zone_forefoot_medial_mean: 0.42,
    zone_forefoot_lateral_mean: 0.28,
    zone_toes_mean: 0.1,
    zone_midfoot_medial_mean: 0.08,
    zone_midfoot_lateral_mean: 0.12,
  },
  fall_risk: {
    gait_speed: 0.72,
    cadence: 88,
    stride_regularity: 0.56,
    step_symmetry: 0.74,
    cop_sway: 0.092,
    ml_index: 0.15,
    arch_index: 0.3,
    acceleration_rms: 1.02,
    zone_heel_medial_mean: 0.2,
    zone_heel_lateral_mean: 0.28,
    zone_forefoot_medial_mean: 0.36,
    zone_forefoot_lateral_mean: 0.34,
    zone_toes_mean: 0.15,
    zone_midfoot_medial_mean: 0.1,
    zone_midfoot_lateral_mean: 0.09,
  },
};

function wave(i: number, phase: number, amp = 1): number {
  return Math.sin(i * 0.18 + phase) * amp;
}

function makeImu(i: number, phase: number, intensity: number): ImuSample {
  return {
    ax: wave(i, phase, intensity),
    ay: wave(i, phase + 1.4, intensity * 0.5),
    az: 0.9 + wave(i, phase + 0.7, intensity * 0.18),
    gx: wave(i, phase + 0.2, intensity * 0.25),
    gy: wave(i, phase + 1.1, intensity * 0.2),
    gz: wave(i, phase + 2.1, intensity * 0.16),
  };
}

function makePressure(i: number, phase: number, asymmetry: number): PressureGrid {
  const load = Math.max(0, Math.sin(i * 0.18 + phase));
  return Array.from({ length: 16 }, (_, row) =>
    Array.from({ length: 8 }, (_, col) => {
      const heel = row > 10 ? 0.9 : 0.35;
      const forefoot = row < 5 ? 0.75 : 0.25;
      const lateral = col > 3 ? 1 + asymmetry : 1 - asymmetry;
      return Number((load * (heel + forefoot) * lateral).toFixed(3));
    }),
  );
}

function makeFrame(i: number, side: "left" | "right", profile: MockProfile): FootSensorFrame {
  const phase = side === "left" ? 0 : Math.PI;
  const asymmetry = profile === "stroke" && side === "left" ? -0.28 : profile === "fall_risk" ? 0.12 : 0.03;
  const intensity = profile === "parkinsons" ? 0.55 : profile === "fall_risk" ? 0.75 : 1;

  return {
    timestampMs: i * 33,
    side,
    imu: makeImu(i, phase, intensity),
    pressure: makePressure(i, phase, asymmetry),
  };
}

function makeSkeleton(frameCount: number): number[][][] {
  return Array.from({ length: frameCount }, (_, frame) =>
    Array.from({ length: 17 }, (_, joint) => [
      Number((joint * 0.04 + wave(frame, joint * 0.1, 0.03)).toFixed(4)),
      Number((joint * 0.08 + wave(frame, joint * 0.2, 0.02)).toFixed(4)),
      Number(wave(frame, joint * 0.3, 0.04).toFixed(4)),
    ]),
  );
}

export function generateMockSensorData(profile: MockProfile = "normal", frameCount = 128): MockSensorData {
  const frames: BilateralSensorFrame[] = Array.from({ length: frameCount }, (_, i) => ({
    timestampMs: i * 33,
    left: makeFrame(i, "left", profile),
    right: makeFrame(i, "right", profile),
  }));

  return {
    meta: {
      sampleRateHz: 30,
      startedAtMs: Date.now(),
      frameCount,
    },
    frames,
    features: PROFILE_FEATURES[profile],
    payload: {
      imu: frames.map((frame) => {
        const imu = frame.left.imu;
        return [imu.ax, imu.ay, imu.az, imu.gx, imu.gy, imu.gz];
      }),
      pressure: frames[Math.floor(frameCount / 2)].left.pressure,
      skeleton: makeSkeleton(frameCount),
    },
  };
}
