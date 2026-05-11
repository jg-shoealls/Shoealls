"use client";

import { useCallback, useState } from "react";
import { ResultCard } from "@/components/ResultCard";
import Sidebar from "@/components/Sidebar";
import { defaultInferenceEngine } from "@/lib/inference";
import { generateMockSensorData, MockProfile, PROFILE_LABELS } from "@/lib/mockSensorData";
import type { InferenceOutput } from "@/types/sensor";

const PROFILES: MockProfile[] = ["normal", "parkinsons", "stroke", "fall_risk"];
const COLORS = { purple: "#AF65FA", amber: "#F59E0B" };

export default function ReasoningPage() {
  const [profile, setProfile] = useState<MockProfile>("fall_risk");
  const [result, setResult] = useState<InferenceOutput | null>(null);
  const [loading, setLoading] = useState(false);

  const run = useCallback(async () => {
    setLoading(true);
    const mock = generateMockSensorData(profile);
    setResult(await defaultInferenceEngine.infer({ sensorData: mock.payload, features: mock.features }));
    setLoading(false);
  }, [profile]);

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-textPri font-semibold text-xl">AI 추론 인터페이스</h1>
            <p className="text-textMuted text-[12px] mt-0.5">현재는 룰 기반 추론이며 이후 LSTM/Transformer adapter로 교체 가능합니다.</p>
          </div>
          <div className="flex items-center gap-3">
            <select value={profile} onChange={(event) => setProfile(event.target.value as MockProfile)} className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5">
              {PROFILES.map((item) => <option key={item} value={item}>{PROFILE_LABELS[item]}</option>)}
            </select>
            <button onClick={run} disabled={loading} className="bg-purple hover:bg-purple/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg">
              {loading ? "추론 중" : "추론 실행"}
            </button>
          </div>
        </header>
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {result ? (
            <ResultCard title="추론 결과" badge={result.modelName} accentColor={COLORS.purple} isDemo={result.isMock}>
              <div className="grid grid-cols-4 gap-3">
                {[
                  ["보행 점수", result.report.score],
                  ["대칭성", result.report.symmetryPct],
                  ["리듬", result.report.rhythmPct],
                  ["안정성", result.report.stabilityPct],
                ].map(([label, value]) => (
                  <div key={label} className="bg-surface rounded-lg p-3">
                    <div className="text-textMuted text-[11px]">{label}</div>
                    <div className="text-textPri text-xl font-bold mt-1">{value}</div>
                  </div>
                ))}
              </div>
              <div className="mt-4 text-textSec text-[13px]">
                낙상 위험: {result.report.fallRisk.label} · {result.report.fallRisk.recommendation}
              </div>
            </ResultCard>
          ) : (
            <div className="bg-card rounded-xl p-10 text-center text-textMuted text-[14px]">추론을 실행해 adapter 결과를 확인하세요.</div>
          )}
        </div>
      </main>
    </div>
  );
}
