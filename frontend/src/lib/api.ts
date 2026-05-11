import type { GaitFeatures, MultimodalSensorPayload } from "@/types/sensor";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

export type SensorData = MultimodalSensorPayload;

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
  ml_top3: { name_kr: string; probability: number }[];
  abnormal_biomarkers: string[];
}

export interface InjuryRiskResponse {
  predicted_injury: string;
  predicted_injury_kr: string;
  confidence: number;
  top3: { name_kr: string; probability: number }[];
  combined_risk_score: number;
  combined_risk_grade: string;
  timeline: string;
  priority_actions: string[];
  body_risk_map: Record<string, number>;
}

export interface ReasoningStep {
  step: number;
  label: string;
  prediction: string;
  prediction_kr: string;
  probability: number;
}

export interface ReasoningResponse {
  final_prediction: string;
  final_prediction_kr: string;
  confidence: number;
  reasoning_trace: ReasoningStep[];
  anomaly_findings: Array<{ modality: string; anomalies: Array<{ type: string; score: number }> }>;
  modality_weights: Record<string, number>;
  uncertainty: number;
  evidence_strength: number;
  report_kr: string;
  is_demo_mode: boolean;
}

export interface AnalyzeResponse {
  classify: ClassifyResponse;
  disease_risk: DiseaseRiskResponse;
  injury_risk: InjuryRiskResponse;
  reasoning: ReasoningResponse;
}

export interface SampleResponse {
  gait_profile: string;
  sensor_data: SensorData;
  features: GaitFeatures;
}

async function apiFetch<T>(path: string, options?: RequestInit, apiKey?: string): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(apiKey ? { "X-API-Key": apiKey } : {}),
  };

  const response = await fetch(`${BASE}${path}`, {
    ...options,
    headers: { ...headers, ...(options?.headers as Record<string, string> | undefined) },
  });

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(
      typeof errorPayload.detail === "string" ? errorPayload.detail : JSON.stringify(errorPayload.detail),
    );
  }

  return response.json() as Promise<T>;
}

export const api = {
  health: () => apiFetch<{ status: string; version: string }>("/health"),

  sample: (profile = "normal") => apiFetch<SampleResponse>(`/api/v1/sample?gait_profile=${profile}`),

  classify: (sensorData: SensorData, apiKey?: string) =>
    apiFetch<ClassifyResponse>(
      "/api/v1/classify",
      { method: "POST", body: JSON.stringify({ sensor_data: sensorData }) },
      apiKey,
    ),

  analyze: (sensorData: SensorData, features: GaitFeatures, apiKey?: string) =>
    apiFetch<AnalyzeResponse>(
      "/api/v1/analyze",
      { method: "POST", body: JSON.stringify({ sensor_data: sensorData, features }) },
      apiKey,
    ),
};
