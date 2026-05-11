"use client";

import { useCallback, useState } from "react";
import { ProgressBar, ResultCard } from "@/components/ResultCard";
import Sidebar from "@/components/Sidebar";
import { api, DiseaseRiskResponse } from "@/lib/api";
import { generateMockSensorData, MockProfile, PROFILE_LABELS } from "@/lib/mockSensorData";
import { calculateFallRisk } from "@/lib/gaitAnalysis";

const PROFILES: MockProfile[] = ["normal", "parkinsons", "stroke", "fall_risk"];
const COLORS = { green: "#10B981", amber: "#F59E0B", red: "#EF4444" };

export default function DiseasePage() {
  const [profile, setProfile] = useState<MockProfile>("parkinsons");
  const [result, setResult] = useState<DiseaseRiskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const sample = await api.sample(profile);
      const analysis = await api.analyze(sample.sensor_data, sample.features);
      setResult(analysis.disease_risk);
    } catch (err) {
      const mock = generateMockSensorData(profile);
      const risk = calculateFallRisk(mock.features);
      setResult({
        top_diseases: [],
        ml_prediction: "risk_sign",
        ml_prediction_kr: "보행 이상 징후 관찰",
        ml_confidence: risk.score,
        ml_top3: [
          { name_kr: "보행 리듬 저하 징후", probability: mock.features.stride_regularity < 0.65 ? 0.68 : 0.18 },
          { name_kr: "좌우 비대칭 징후", probability: 1 - mock.features.step_symmetry },
          { name_kr: "동적 안정성 저하 징후", probability: Math.min(0.9, mock.features.cop_sway * 8) },
        ],
        abnormal_biomarkers: risk.reasons,
      });
      setError(err instanceof Error ? err.message : "API 연결 실패");
    } finally {
      setLoading(false);
    }
  }, [profile]);

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-textPri font-semibold text-xl">위험 징후 모니터링</h1>
            <p className="text-textMuted text-[12px] mt-0.5">질병 확정 진단이 아닌 보행 기반 관찰 지표입니다.</p>
          </div>
          <div className="flex items-center gap-3">
            <select value={profile} onChange={(event) => setProfile(event.target.value as MockProfile)} className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5">
              {PROFILES.map((item) => <option key={item} value={item}>{PROFILE_LABELS[item]}</option>)}
            </select>
            <button onClick={run} disabled={loading} className="bg-blue hover:bg-blue/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg">
              {loading ? "분석 중" : "징후 분석"}
            </button>
          </div>
        </header>
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {error && <div className="bg-amber/10 border border-amber/30 text-amber rounded-xl px-5 py-4 text-[13px]">API 연결 실패: mock 결과를 표시합니다. {error}</div>}
          {result ? (
            <ResultCard title="보행 이상 징후 요약" badge="risk signs" accentColor={COLORS.green}>
              <div className="text-textPri text-xl font-bold mb-1">{result.ml_prediction_kr}</div>
              <div className="text-textSec text-[13px] mb-4">관찰 강도 {(result.ml_confidence * 100).toFixed(1)}%</div>
              <div className="space-y-3">
                {result.ml_top3.map((item) => (
                  <ProgressBar key={item.name_kr} pct={item.probability} color={item.probability > 0.5 ? COLORS.red : COLORS.amber} label={item.name_kr} valueLabel={`${(item.probability * 100).toFixed(1)}%`} />
                ))}
              </div>
              <p className="text-textMuted text-[11px] mt-4">반복 관찰 시 보호자 공유 및 전문 의료기관 상담을 권고합니다.</p>
            </ResultCard>
          ) : (
            <div className="bg-card rounded-xl p-10 text-center text-textMuted text-[14px]">프로파일을 선택하고 징후 분석을 실행하세요.</div>
          )}
        </div>
      </main>
    </div>
  );
}
