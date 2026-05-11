import { buildGaitReport } from "@/lib/gaitAnalysis";
import { generateMockSensorData } from "@/lib/mockSensorData";
import type { GaitInferenceEngine, InferenceInput, InferenceOutput } from "@/types/sensor";

/**
 * 룰 기반 보행 리포트. 시계열 프레임이 없으면 특성만으로 지표를 채웁니다.
 * (향후 Transformer/LSTM은 동일 InferenceOutput으로 교체)
 */
export class RuleBasedGaitInference implements GaitInferenceEngine {
  async infer(input: InferenceInput): Promise<InferenceOutput> {
    return {
      modelName: "rule-based-v1",
      isMock: false,
      report: buildGaitReport(input.features, []),
    };
  }
}

/** 학습 가중치 없음 — 이벤트 검출용 합성 프레임으로 스텁 추론 */
export class MockGaitInference implements GaitInferenceEngine {
  async infer(input: InferenceInput): Promise<InferenceOutput> {
    const seqLen = input.sensorData.imu.length || 128;
    const mock = generateMockSensorData("fall_risk", seqLen);

    return {
      modelName: "mock-neural-adapter-v0",
      isMock: true,
      report: buildGaitReport(input.features, mock.frames),
    };
  }
}

export type AiModelAdapter = GaitInferenceEngine;

export const defaultInferenceEngine: GaitInferenceEngine = new RuleBasedGaitInference();
