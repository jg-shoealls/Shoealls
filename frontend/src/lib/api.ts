/**
 * Shoealls API 클라이언트
 * FastAPI 백엔드와 통신하는 타입-안전 함수 모음
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

export interface SensorData {
  imu: number[][];       // [128, 6]
  pressure: number[][];  // [16, 8]
  skeleton: number[][][]; // [128, 17, 3]
}

export interface ClassifyResponse {
  prediction: string;
  prediction_kr: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  is_demo_mode: boolean;
}

export interface DiseaseRisk {
  disease: string;
  disease_kr: string;
  risk_score: number;
  severity: string;
  key_signs: string[];
  referral: string;
}

export interface DiseaseRiskResponse {
  top_diseases: DiseaseRisk[];
  ml_prediction: string;
  ml_prediction_kr: string;
  ml_confidence: number;
  ml_top3: { disease: string; disease_kr: string; probability: number }[];
  anomaly_biomarkers: string[];
}

export interface InjuryRiskResponse {
  overall_risk_score: number;
  risk_level: string;
  risk_level_kr: string;
  top_injuries: { injury: string; injury_kr: string; probability: number }[];
  injury_timeline: string;
  prevention_advice: string[];
}

export interface ReasoningStep {
  step: number;
  label: string;
  prediction: string;
  prediction_kr: string;
  confidence: number;
}

export interface ReasoningResponse {
  final_prediction: string;
  final_prediction_kr: string;
  final_confidence: number;
  reasoning_trace: ReasoningStep[];
  anomaly_findings: Record<string, string[]>;
  uncertainty: number;
  evidence_strength: number;
  report_kr: string;
}

export interface AnalyzeResponse {
  classification: ClassifyResponse;
  disease_risk: DiseaseRiskResponse;
  injury_risk: InjuryRiskResponse;
  reasoning: ReasoningResponse;
}

export interface SampleResponse {
  gait_profile: string;
  sensor_data: SensorData;
  features: Record<string, number>;
}

async function apiFetch<T>(
  path: string,
  options?: RequestInit,
  apiKey?: string
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(apiKey ? { "X-API-Key": apiKey } : {}),
  };

  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: { ...headers, ...(options?.headers as Record<string, string> ?? {}) },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof err.detail === "string"
        ? err.detail
        : JSON.stringify(err.detail)
    );
  }
  return res.json();
}

export const api = {
  health: () => apiFetch<{ status: string; version: string }>("/health"),

  sample: (profile = "normal") =>
    apiFetch<SampleResponse>(`/api/v1/sample?gait_profile=${profile}`),

  classify: (sensorData: SensorData, apiKey?: string) =>
    apiFetch<ClassifyResponse>(
      "/api/v1/classify",
      { method: "POST", body: JSON.stringify({ sensor_data: sensorData }) },
      apiKey
    ),

  analyze: (sensorData: SensorData, features: Record<string, number>, apiKey?: string) =>
    apiFetch<AnalyzeResponse>(
      "/api/v1/analyze",
      { method: "POST", body: JSON.stringify({ sensor_data: sensorData, features }) },
      apiKey
    ),
};
