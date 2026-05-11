"use client";

import { useCallback, useState } from "react";
import { FallRiskAlert } from "@/components/FallRiskAlert";
import { ProgressBar, ResultCard } from "@/components/ResultCard";
import Sidebar from "@/components/Sidebar";
import { api, InjuryRiskResponse } from "@/lib/api";
import { buildGaitReport } from "@/lib/gaitAnalysis";
import { generateMockSensorData, MockProfile, PROFILE_LABELS } from "@/lib/mockSensorData";
import type { GaitReport } from "@/types/sensor";

const PROFILES: MockProfile[] = ["normal", "parkinsons", "stroke", "fall_risk"];
const COLORS = { amber: "#F59E0B", red: "#EF4444" };

export default function InjuryPage() {
  const [profile, setProfile] = useState<MockProfile>("fall_risk");
  const [report, setReport] = useState<GaitReport>(() => {
    const mock = generateMockSensorData("fall_risk");
    return buildGaitReport(mock.features, mock.frames);
  });
  const [apiRisk, setApiRisk] = useState<InjuryRiskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    const mock = generateMockSensorData(profile);
    setReport(buildGaitReport(mock.features, mock.frames));
    try {
      const sample = await api.sample(profile);
      const analysis = await api.analyze(sample.sensor_data, sample.features);
      setApiRisk(analysis.injury_risk);
    } catch (err) {
      setApiRisk(null);
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
          <h1 className="text-textPri font-semibold text-xl">낙상/부상 위험 모니터링</h1>
          <div className="flex items-center gap-3">
            <select value={profile} onChange={(event) => setProfile(event.target.value as MockProfile)} className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5">
              {PROFILES.map((item) => <option key={item} value={item}>{PROFILE_LABELS[item]}</option>)}
            </select>
            <button onClick={run} disabled={loading} className="bg-blue hover:bg-blue/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg">
              {loading ? "분석 중" : "위험 평가"}
            </button>
          </div>
        </header>
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {error && <div className="bg-amber/10 border border-amber/30 text-amber rounded-xl px-5 py-4 text-[13px]">API 연결 실패: 로컬 룰 기반 결과를 표시합니다. {error}</div>}
          <FallRiskAlert risk={report.fallRisk} />
          {apiRisk && (
            <ResultCard title="백엔드 낙상/부상 위험 결과" badge="API risk" accentColor={COLORS.amber}>
              <div className="space-y-3">
                <ProgressBar pct={apiRisk.combined_risk_score} color={apiRisk.combined_risk_score > 0.55 ? COLORS.red : COLORS.amber} label="복합 위험도" valueLabel={`${(apiRisk.combined_risk_score * 100).toFixed(1)}%`} />
                {apiRisk.top3.map((item) => (
                  <ProgressBar key={item.name_kr} pct={item.probability} color={item.probability > 0.45 ? COLORS.red : COLORS.amber} label={item.name_kr} valueLabel={`${(item.probability * 100).toFixed(1)}%`} />
                ))}
              </div>
            </ResultCard>
          )}
        </div>
      </main>
    </div>
  );
}
