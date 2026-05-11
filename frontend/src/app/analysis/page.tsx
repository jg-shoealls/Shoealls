"use client";

import { useCallback, useState } from "react";
import { ProgressBar, ResultCard } from "@/components/ResultCard";
import Sidebar from "@/components/Sidebar";
import { api, ClassifyResponse } from "@/lib/api";
import { generateMockSensorData, MockProfile, PROFILE_LABELS } from "@/lib/mockSensorData";

const PROFILES: MockProfile[] = ["normal", "parkinsons", "stroke", "fall_risk"];
const COLORS = { blue: "#3B82F6", green: "#10B981", amber: "#F59E0B" };
const CLASS_LABELS: Record<string, string> = {
  normal: "안정 보행",
  antalgic: "통증 회피 보행",
  ataxic: "불안정 보행",
  parkinsonian: "짧은 보폭 패턴",
};

export default function AnalysisPage() {
  const [profile, setProfile] = useState<MockProfile>("normal");
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const sample = await api.sample(profile);
      setResult(await api.classify(sample.sensor_data));
    } catch (err) {
      const mock = generateMockSensorData(profile);
      setResult({
        prediction: profile === "normal" ? "normal" : "ataxic",
        prediction_kr: profile === "normal" ? "안정 보행" : "불안정 보행",
        confidence: profile === "normal" ? 0.86 : 0.72,
        class_probabilities: { normal: profile === "normal" ? 0.86 : 0.12, antalgic: 0.12, ataxic: profile === "normal" ? 0.02 : 0.72, parkinsonian: 0.04 },
        is_demo_mode: true,
      });
      setError(err instanceof Error ? err.message : "API 연결 실패");
      void mock;
    } finally {
      setLoading(false);
    }
  }, [profile]);

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <h1 className="text-textPri font-semibold text-xl">보행 패턴 분석</h1>
          <div className="flex items-center gap-3">
            <select value={profile} onChange={(event) => setProfile(event.target.value as MockProfile)} className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5">
              {PROFILES.map((item) => <option key={item} value={item}>{PROFILE_LABELS[item]}</option>)}
            </select>
            <button onClick={run} disabled={loading} className="bg-blue hover:bg-blue/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg">
              {loading ? "분석 중" : "분류 실행"}
            </button>
          </div>
        </header>
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {error && <div className="bg-amber/10 border border-amber/30 text-amber rounded-xl px-5 py-4 text-[13px]">API 연결 실패: mock 결과를 표시합니다. {error}</div>}
          {result ? (
            <ResultCard title="보행 분류 결과" badge="classification" accentColor={COLORS.blue} isDemo={result.is_demo_mode}>
              <div className="space-y-3">
                <div className="text-textPri text-2xl font-bold">{CLASS_LABELS[result.prediction] ?? result.prediction_kr}</div>
                <div className="text-textSec text-[13px]">신뢰도 {(result.confidence * 100).toFixed(1)}%</div>
                {Object.entries(result.class_probabilities).map(([name, probability]) => (
                  <ProgressBar key={name} pct={probability} color={COLORS.blue} label={CLASS_LABELS[name] ?? name} valueLabel={`${(probability * 100).toFixed(1)}%`} />
                ))}
              </div>
            </ResultCard>
          ) : (
            <div className="bg-card rounded-xl p-10 text-center text-textMuted text-[14px]">프로파일을 선택하고 분류를 실행하세요.</div>
          )}
        </div>
      </main>
    </div>
  );
}
