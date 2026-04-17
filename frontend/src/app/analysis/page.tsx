"use client";

import { useState, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import { ResultCard, ProgressBar } from "@/components/ResultCard";
import { api, SampleResponse, ClassifyResponse } from "@/lib/api";

const PROFILES = ["normal", "parkinsons", "stroke", "fall_risk"] as const;
type Profile = (typeof PROFILES)[number];

const PROFILE_KR: Record<Profile, string> = {
  normal: "정상 보행",
  parkinsons: "파킨슨",
  stroke: "뇌졸중",
  fall_risk: "낙상 위험",
};

const CLASS_KR: Record<string, string> = {
  normal: "정상 보행",
  antalgic: "절뚝거림",
  ataxic: "운동실조",
  parkinsonian: "파킨슨",
};

const C = {
  blue: "#3B82F6",
  green: "#10B981",
  amber: "#F59E0B",
  red: "#EF4444",
  purple: "#AF65FA",
};

const SENSOR_CARDS = [
  { key: "imu",      title: "IMU",          sub: "가속도 + 자이로스코프 6ch", color: C.blue },
  { key: "pressure", title: "족저압",        sub: "16 × 8 그리드",             color: C.green },
  { key: "skeleton", title: "스켈레톤",      sub: "17 관절 × 3D 좌표",          color: C.purple },
];

const FEATURE_LABELS: Record<string, string> = {
  gait_speed:                "보행 속도 (m/s)",
  cadence:                   "케이던스 (steps/min)",
  stride_regularity:         "보폭 규칙성",
  step_symmetry:             "스텝 대칭성",
  heel_pressure_ratio:       "뒤꿈치 압력 비율",
  forefoot_pressure_ratio:   "앞발 압력 비율",
  arch_index:                "아치 지수",
  pressure_asymmetry:        "압력 비대칭",
  cop_sway:                  "CoP 흔들림",
  ml_variability:            "ML 변동성",
  trunk_sway:                "몸통 흔들림 (deg/s)",
  acceleration_rms:          "가속도 RMS (m/s²)",
  acceleration_variability:  "가속도 변동성",
};

export default function AnalysisPage() {
  const [profile, setProfile] = useState<Profile>("normal");
  const [sample, setSample] = useState<SampleResponse | null>(null);
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await api.sample(profile);
      setSample(s);
      const r = await api.classify(s.sensor_data);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "분석 실패");
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
            <select
              value={profile}
              onChange={(e) => setProfile(e.target.value as Profile)}
              className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue"
            >
              {PROFILES.map((p) => <option key={p} value={p}>{PROFILE_KR[p]}</option>)}
            </select>
            <button
              onClick={run}
              disabled={loading}
              className="bg-blue hover:bg-blue/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg transition-colors"
            >
              {loading ? "분석 중…" : "분류 실행"}
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {error && (
            <div className="bg-red/10 border border-red/30 text-red rounded-xl px-5 py-4 text-[14px]">
              오류: {error}
            </div>
          )}

          {/* 센서 입력 */}
          <section>
            <h2 className="text-textSec text-[14px] font-semibold mb-3">멀티모달 센서 입력</h2>
            <div className="grid grid-cols-3 gap-4">
              {SENSOR_CARDS.map((c) => (
                <div key={c.key} className="bg-card rounded-xl p-5 flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: c.color }} />
                    <span className="text-textPri font-semibold text-[14px]">{c.title}</span>
                  </div>
                  <span className="text-textSec text-[12px]">{c.sub}</span>
                  <span className="text-[12px] font-semibold mt-1" style={{ color: c.color }}>
                    {sample ? "데이터 로드됨" : "대기 중"}
                  </span>
                </div>
              ))}
            </div>
          </section>

          {/* 추출 특성 */}
          {sample && (
            <section>
              <h2 className="text-textSec text-[14px] font-semibold mb-3">추출된 보행 지표 (13개)</h2>
              <div className="bg-card rounded-xl p-5 grid grid-cols-2 gap-x-10 gap-y-3">
                {Object.entries(sample.features).map(([k, v]) => (
                  <div key={k} className="flex justify-between items-center border-b border-border/40 pb-2">
                    <span className="text-textSec text-[12px]">{FEATURE_LABELS[k] ?? k}</span>
                    <span className="text-textPri font-semibold text-[13px]">{v.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 분류 결과 */}
          {loading && (
            <div className="bg-card rounded-2xl p-12 text-center">
              <div className="inline-block w-8 h-8 border-2 border-blue border-t-transparent rounded-full animate-spin mb-3" />
              <div className="text-textSec text-[14px]">보행 패턴 분류 중…</div>
            </div>
          )}

          {result && (
            <ResultCard title="보행 패턴 분류 결과" badge="보행 분류" accentColor={C.blue} isDemo={result.is_demo_mode}>
              <div className="flex items-start gap-6">
                <div className="w-32 h-32 shrink-0 rounded-full bg-surface flex flex-col items-center justify-center border-2 border-blue/40">
                  <span className="text-blue text-[12px] font-medium text-center px-2">{result.prediction_kr}</span>
                  <span className="text-textPri font-bold text-2xl">
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex-1 space-y-3">
                  <div className="text-textSec text-[12px] mb-1">클래스별 확률 분포</div>
                  {Object.entries(result.class_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, prob]) => (
                      <ProgressBar
                        key={name}
                        pct={prob}
                        color={C.blue}
                        label={CLASS_KR[name] ?? name}
                        valueLabel={(prob * 100).toFixed(1) + "%"}
                      />
                    ))}
                </div>
              </div>
            </ResultCard>
          )}

          {!result && !loading && (
            <div className="bg-card rounded-2xl p-12 text-center">
              <div className="text-textMuted text-[14px]">
                프로파일을 선택하고 <strong className="text-textSec">분류 실행</strong>을 눌러 보행 패턴을 분석하세요.
              </div>
            </div>
          )}
        </div>

        <footer className="bg-surface border-t border-border px-8 py-2.5 text-textMuted text-[11px] flex justify-between shrink-0">
          <span>POST /api/v1/classify · 4-class gait classifier</span>
          <span>© 2026 Shoealls</span>
        </footer>
      </main>
    </div>
  );
}
