"use client";

import { useCallback, useMemo, useState } from "react";
import { FallRiskAlert } from "@/components/FallRiskAlert";
import { GaitReportCard } from "@/components/GaitReportCard";
import { ProgressBar, ResultCard } from "@/components/ResultCard";
import Sidebar from "@/components/Sidebar";
import GaitTrendChart from "@/components/GaitTrendChart";
import RealtimeWaveform from "@/components/RealtimeWaveform";
import PressureHeatmap from "@/components/PressureHeatmap";
import { buildGaitReport } from "@/lib/gaitAnalysis";
import { generateMockSensorData, MockProfile, PROFILE_LABELS } from "@/lib/mockSensorData";
import { api, AnalyzeResponse } from "@/lib/api";
import type { GaitReport, MockSensorData } from "@/types/sensor";

const PROFILES: MockProfile[] = ["normal", "parkinsons", "stroke", "fall_risk"];

const COLORS = {
  blue: "#3B82F6", green: "#10B981", amber: "#F59E0B",
  red: "#EF4444", purple: "#AF65FA",
};

const CLASS_LABELS: Record<string, string> = {
  normal: "안정 보행",
  antalgic: "통증 회피 보행",
  ataxic: "불안정 보행",
  parkinsonian: "짧은 보폭 패턴",
};

const TREND_DATA: Record<MockProfile, { date: string; speed: number; stability: number }[]> = {
  normal: [
    { date: "5/5", speed: 1.18, stability: 88 }, { date: "5/6", speed: 1.20, stability: 90 },
    { date: "5/7", speed: 1.22, stability: 89 }, { date: "5/8", speed: 1.25, stability: 91 },
    { date: "5/9", speed: 1.21, stability: 90 }, { date: "5/10", speed: 1.23, stability: 92 },
    { date: "5/11", speed: 1.22, stability: 91 },
  ],
  parkinsons: [
    { date: "5/5", speed: 0.82, stability: 72 }, { date: "5/6", speed: 0.79, stability: 69 },
    { date: "5/7", speed: 0.75, stability: 65 }, { date: "5/8", speed: 0.71, stability: 62 },
    { date: "5/9", speed: 0.70, stability: 58 }, { date: "5/10", speed: 0.69, stability: 55 },
    { date: "5/11", speed: 0.68, stability: 52 },
  ],
  stroke: [
    { date: "5/5", speed: 0.60, stability: 55 }, { date: "5/6", speed: 0.58, stability: 53 },
    { date: "5/7", speed: 0.55, stability: 51 }, { date: "5/8", speed: 0.54, stability: 50 },
    { date: "5/9", speed: 0.53, stability: 48 }, { date: "5/10", speed: 0.53, stability: 47 },
    { date: "5/11", speed: 0.52, stability: 46 },
  ],
  fall_risk: [
    { date: "5/5", speed: 0.85, stability: 68 }, { date: "5/6", speed: 0.81, stability: 65 },
    { date: "5/7", speed: 0.78, stability: 62 }, { date: "5/8", speed: 0.76, stability: 60 },
    { date: "5/9", speed: 0.74, stability: 57 }, { date: "5/10", speed: 0.73, stability: 55 },
    { date: "5/11", speed: 0.72, stability: 53 },
  ],
};

function buildLocalAnalysis(profile: MockProfile): { sample: MockSensorData; report: GaitReport } {
  const sample = generateMockSensorData(profile);
  return { sample, report: buildGaitReport(sample.features, sample.frames) };
}

export default function Dashboard() {
  const [profile, setProfile] = useState<MockProfile>("parkinsons");
  const [sample, setSample] = useState<MockSensorData>(() => buildLocalAnalysis("parkinsons").sample);
  const [report, setReport] = useState<GaitReport>(() => buildLocalAnalysis("parkinsons").report);
  const [apiResult, setApiResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<"local" | "api">("local");

  const keyStats = useMemo(
    () => [
      { label: "보행 속도", value: `${sample.features.gait_speed.toFixed(2)} m/s`, color: COLORS.blue },
      { label: "케이던스", value: `${Math.round(sample.features.cadence)} spm`, color: COLORS.green },
      { label: "좌우 대칭", value: `${Math.round(sample.features.step_symmetry * 100)}%`, color: COLORS.purple },
      { label: "CoP 흔들림", value: sample.features.cop_sway.toFixed(3), color: COLORS.amber },
    ],
    [sample],
  );

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    setError(null);
    const local = buildLocalAnalysis(profile);
    setSample(local.sample);
    setReport(local.report);
    setSource("local");
    try {
      const backendSample = await api.sample(profile);
      const backendResult = await api.analyze(backendSample.sensor_data, backendSample.features);
      setApiResult(backendResult);
      setSource("api");
    } catch (err) {
      setApiResult(null);
      setError(err instanceof Error ? err.message : "API 분석 실패 — 로컬 룰 기반 결과 표시 중");
    } finally {
      setLoading(false);
    }
  }, [profile]);

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* ── Header ── */}
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-textPri font-semibold text-xl">스마트슈즈 헬스케어 대시보드</h1>
            <p className="text-textMuted text-[12px] mt-0.5">IMU · 족저압 · 스켈레톤 멀티모달 실시간 모니터링</p>
          </div>
          <div className="flex items-center gap-3">
            <select
              value={profile}
              onChange={(e) => setProfile(e.target.value as MockProfile)}
              className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue"
            >
              {PROFILES.map((p) => (
                <option key={p} value={p}>{PROFILE_LABELS[p]}</option>
              ))}
            </select>
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium text-blue bg-blue/20">
              {source === "api" ? "FastAPI" : "Local rule"}
            </span>
            <button
              onClick={runAnalysis}
              disabled={loading}
              className="bg-blue hover:bg-blue/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg transition-colors"
            >
              {loading ? "분석 중…" : "AI 분석 실행"}
            </button>
          </div>
        </header>

        {/* ── Body ── */}
        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {error && (
            <div className="bg-amber/10 border border-amber/30 text-amber rounded-xl px-5 py-3 text-[12px]">
              {error}
            </div>
          )}

          <FallRiskAlert risk={report.fallRisk} />

          {/* Key stats */}
          <section className="grid grid-cols-4 gap-4">
            {keyStats.map((stat) => (
              <div key={stat.label} className="bg-card rounded-xl p-4 border border-border/50">
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: stat.color }} />
                  <span className="text-textSec text-[12px]">{stat.label}</span>
                </div>
                <div className="text-textPri text-[22px] font-bold">{stat.value}</div>
              </div>
            ))}
          </section>

          {/* Gait report + sensor status */}
          <section className="grid grid-cols-2 gap-5">
            <GaitReportCard report={report} />
            <ResultCard title="센서 입력 상태" badge="BLE · Live" accentColor={COLORS.green}>
              <div className="grid grid-cols-3 gap-3 mb-4">
                {[
                  { label: "IMU",    value: `${sample.payload.imu.length} frames`, sub: "6-axis" },
                  { label: "압력센서", value: "16 × 8",                              sub: "양발" },
                  { label: "샘플링",  value: `${sample.meta.sampleRateHz} Hz`,       sub: "time series" },
                ].map((item) => (
                  <div key={item.label} className="bg-surface rounded-lg p-3">
                    <div className="text-textMuted text-[11px]">{item.label}</div>
                    <div className="text-textPri font-semibold text-[15px] mt-1">{item.value}</div>
                    <div className="text-textMuted text-[10px] mt-1">{item.sub}</div>
                  </div>
                ))}
              </div>
              <ProgressBar pct={report.symmetryPct / 100} color={COLORS.blue}   label="좌우 데이터 동기화"  valueLabel={`${report.symmetryPct}%`} />
              <div className="mt-2" />
              <ProgressBar pct={report.weightTransferPct / 100} color={COLORS.green} label="체중 이동 안정성" valueLabel={`${report.weightTransferPct}%`} />
            </ResultCard>
          </section>

          {/* Realtime waveform + pressure heatmap */}
          <section className="grid grid-cols-2 gap-5">
            <RealtimeWaveform profile={profile} />
            <PressureHeatmap profile={profile} />
          </section>

          {/* 7-day trend */}
          <section>
            <GaitTrendChart data={TREND_DATA[profile]} />
          </section>

          {/* Backend AI results */}
          {apiResult && (
            <section className="grid grid-cols-2 gap-5">
              <ResultCard title="AI 보행 분류" badge="API inference" accentColor={COLORS.purple} isDemo={apiResult.classify.is_demo_mode}>
                <div className="flex items-start gap-5">
                  <div className="w-28 h-28 shrink-0 rounded-full bg-surface flex flex-col items-center justify-center border border-purple/40">
                    <span className="text-purple text-[12px] text-center px-2 leading-tight">
                      {CLASS_LABELS[apiResult.classify.prediction] ?? apiResult.classify.prediction_kr}
                    </span>
                    <span className="text-textPri font-bold text-xl mt-1">
                      {(apiResult.classify.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex-1 space-y-3">
                    {Object.entries(apiResult.classify.class_probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([name, prob]) => (
                        <ProgressBar key={name} pct={prob} color={COLORS.purple}
                          label={CLASS_LABELS[name] ?? name} valueLabel={`${(prob * 100).toFixed(1)}%`} />
                      ))}
                  </div>
                </div>
              </ResultCard>

              <ResultCard title="위험 징후 요약" badge="monitoring" accentColor={COLORS.amber}>
                <div className="space-y-3">
                  <ProgressBar
                    pct={apiResult.injury_risk.combined_risk_score}
                    color={apiResult.injury_risk.combined_risk_score > 0.55 ? COLORS.red : COLORS.amber}
                    label="낙상/부상 복합 위험"
                    valueLabel={`${(apiResult.injury_risk.combined_risk_score * 100).toFixed(1)}%`}
                  />
                  {apiResult.disease_risk.abnormal_biomarkers.slice(0, 4).map((marker) => (
                    <div key={marker} className="text-textSec text-[12px] bg-surface rounded-md px-3 py-2">{marker}</div>
                  ))}
                  <p className="text-textMuted text-[11px] leading-relaxed">
                    보행 이상 징후 표시이며 진단이 아닙니다. 반복 관찰 시 전문 의료기관 방문을 권고합니다.
                  </p>
                </div>
              </ResultCard>
            </section>
          )}
        </div>

        <footer className="bg-surface border-t border-border px-8 py-2.5 text-textMuted text-[11px] flex justify-between shrink-0">
          <span>Shoealls · IMU + Pressure + Skeleton · BiLSTM + Transformer Ensemble</span>
          <span>© 2026 Shoealls Healthcare</span>
        </footer>
      </main>
    </div>
  );
}
