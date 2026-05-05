"use client";

import { useState, useCallback, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import { ResultCard, ProgressBar } from "@/components/ResultCard";
import ThemeToggle from "@/components/ThemeToggle";
import RiskGauge from "@/components/RiskGauge";
import ExportButton from "@/components/ExportButton";
import { api, AnalyzeResponse, SampleResponse } from "@/lib/api";
import {
  saveSession,
  getSessions,
  computeBaseline,
  deviationZ,
  type GaitSession,
  type BaselineStats,
} from "@/lib/baseline";

const PROFILES = ["normal", "parkinsons", "stroke", "fall_risk"] as const;
type Profile = (typeof PROFILES)[number];

const PROFILE_KR: Record<Profile, string> = {
  normal: "정상 보행",
  parkinsons: "파킨슨",
  stroke: "뇌졸중",
  fall_risk: "낙상 위험",
};

const C = {
  blue:   "#3B82F6",
  green:  "#10B981",
  amber:  "#F59E0B",
  red:    "#EF4444",
  purple: "#AF65FA",
};

const CLASS_KR: Record<string, string> = {
  normal:       "정상 보행",
  antalgic:     "절뚝거림",
  ataxic:       "운동실조",
  parkinsonian: "파킨슨",
};

const FEATURE_LABELS: Record<string, string> = {
  gait_speed:               "보행 속도",
  cadence:                  "케이던스",
  stride_regularity:        "보폭 규칙성",
  step_symmetry:            "스텝 대칭성",
  cop_sway:                 "CoP 흔들림",
  ml_variability:           "ML 변동성",
  heel_pressure_ratio:      "뒤꿈치 압력",
  forefoot_pressure_ratio:  "앞발 압력",
  arch_index:               "아치 지수",
  pressure_asymmetry:       "압력 비대칭",
  acceleration_rms:         "가속도 RMS",
  acceleration_variability: "가속도 변동성",
  trunk_sway:               "몸통 흔들림",
};

// Features where higher is worse (inverted deviation coloring)
const LOWER_BETTER = new Set([
  "cop_sway", "ml_variability", "pressure_asymmetry",
  "acceleration_variability", "trunk_sway",
]);

function zColor(z: number | null, key: string): string {
  if (z == null) return C.blue;
  const abs = Math.abs(z);
  const isHigherWorse = LOWER_BETTER.has(key);
  const bad = isHigherWorse ? z > 0 : z < 0;
  if (abs < 1) return C.green;
  if (abs < 2) return bad ? C.amber : C.green;
  return bad ? C.red : C.blue;
}

export default function Dashboard() {
  const [profile, setProfile] = useState<Profile>("parkinsons");
  const [sample, setSample] = useState<SampleResponse | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [baseline, setBaseline] = useState<BaselineStats | null>(null);
  const [sessionCount, setSessionCount] = useState(0);

  useEffect(() => {
    const sessions = getSessions();
    setSessionCount(sessions.length);
    setBaseline(computeBaseline(sessions));
  }, []);

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await api.sample(profile);
      setSample(s);
      const r = await api.analyze(s.sensor_data, s.features);
      setResult(r);

      const session: GaitSession = {
        id: Math.random().toString(36).slice(2, 10),
        timestamp: new Date().toLocaleString("ko-KR"),
        profile,
        features: s.features,
        injuryRisk: r.injury_risk.combined_risk_score,
      };
      saveSession(session);
      const sessions = getSessions();
      setSessionCount(sessions.length);
      setBaseline(computeBaseline(sessions));
    } catch (e) {
      setError(e instanceof Error ? e.message : "분석 실패");
    } finally {
      setLoading(false);
    }
  }, [profile]);

  const cls = result?.classify;
  const dis = result?.disease_risk;
  const inj = result?.injury_risk;
  const rsn = result?.reasoning;

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* 탑바 */}
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <h1 className="text-textPri font-semibold text-xl">보행 분석 대시보드</h1>
          <div className="flex items-center gap-3">
            <select
              value={profile}
              onChange={(e) => setProfile(e.target.value as Profile)}
              className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue"
            >
              {PROFILES.map((p) => (
                <option key={p} value={p}>{PROFILE_KR[p]}</option>
              ))}
            </select>
            {cls?.is_demo_mode !== false && (
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium text-amber bg-amber/20">
                데모 모드
              </span>
            )}
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium text-blue bg-blue/20">
              v0.1.0 MVP
            </span>
            <ExportButton result={result} profile={PROFILE_KR[profile]} />
            <ThemeToggle />
            <button
              onClick={runAnalysis}
              disabled={loading}
              className="bg-blue hover:bg-blue/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg transition-colors"
            >
              {loading ? "분석 중…" : "분석 시작"}
            </button>
          </div>
        </header>

        {/* 스크롤 영역 */}
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {error && (
            <div className="bg-red/10 border border-red/30 text-red rounded-xl px-5 py-4 text-[14px]">
              오류: {error}
            </div>
          )}

          {/* 센서 입력 요약 */}
          <section>
            <h2 className="text-textSec text-[14px] font-semibold mb-3">센서 입력 데이터</h2>
            <div className="grid grid-cols-3 gap-4">
              {[
                { title: "IMU 센서", sub: "가속도 + 자이로스코프 6ch", val: "128 프레임", color: C.blue },
                { title: "족저압 센서", sub: "16 × 8 그리드", val: sample ? "로드됨" : "대기 중", color: C.green },
                { title: "스켈레톤", sub: "17 관절 × 3D 좌표", val: "128 프레임", color: C.purple },
              ].map((card) => (
                <div key={card.title} className="bg-card rounded-xl p-4 flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: card.color }} />
                    <span className="text-textPri font-semibold text-[14px]">{card.title}</span>
                  </div>
                  <span className="text-textSec text-[11px]">{card.sub}</span>
                  <span className="text-[13px] font-semibold mt-1" style={{ color: card.color }}>
                    {card.val}
                  </span>
                </div>
              ))}
            </div>
          </section>

          {/* 초기 안내 */}
          {!result && !loading && (
            <div className="bg-card rounded-2xl p-12 text-center">
              <div className="text-textMuted text-[14px]">
                보행 프로파일을 선택하고 <strong className="text-textSec">분석 시작</strong>을 누르세요.
              </div>
              {sessionCount > 0 && (
                <div className="mt-2 text-textMuted text-[12px]">
                  저장된 세션 {sessionCount}건
                  {baseline ? ` · 개인 기준선 활성 (${baseline.sessions}회 기준)` : " · 기준선 구축까지 " + (3 - sessionCount) + "회 필요"}
                </div>
              )}
            </div>
          )}

          {/* 로딩 */}
          {loading && (
            <div className="bg-card rounded-2xl p-12 text-center">
              <div className="inline-block w-8 h-8 border-2 border-blue border-t-transparent rounded-full animate-spin mb-3" />
              <div className="text-textSec text-[14px]">AI 분석 중…</div>
            </div>
          )}

          {/* 결과 카드 그리드 */}
          {result && (
            <>
              <h2 className="text-textSec text-[14px] font-semibold">분석 결과</h2>
              <div className="grid grid-cols-2 gap-5">

                {/* ── 보행 패턴 분류 ── */}
                <ResultCard title="보행 패턴 분류" badge="보행 분류" accentColor={C.blue} isDemo={cls?.is_demo_mode}>
                  <div className="flex items-start gap-5">
                    <div className="w-28 h-28 shrink-0 rounded-full bg-surface flex flex-col items-center justify-center border-2 border-blue/40">
                      <span className="text-blue text-[11px] font-medium text-center px-1">{cls?.prediction_kr}</span>
                      <span className="text-textPri font-bold text-xl">
                        {cls ? (cls.confidence * 100).toFixed(1) + "%" : "--"}
                      </span>
                    </div>
                    <div className="flex-1 space-y-3">
                      {cls && Object.entries(cls.class_probabilities)
                        .sort((a, b) => b[1] - a[1])
                        .map(([name, prob]) => (
                          <ProgressBar key={name} pct={prob} color={C.blue}
                            label={CLASS_KR[name] ?? name}
                            valueLabel={(prob * 100).toFixed(1) + "%"} />
                        ))}
                    </div>
                  </div>
                </ResultCard>

                {/* ── 질환 위험도 ── */}
                <ResultCard title="질환 위험 예측" badge="질환 위험도" accentColor={C.green}>
                  {dis && (
                    <>
                      <div className="text-green text-[13px] font-semibold mb-3">
                        ML 예측: {dis.ml_prediction_kr}  {(dis.ml_confidence * 100).toFixed(1)}%
                      </div>
                      <div className="space-y-3">
                        {dis.ml_top3.map((d) => (
                          <ProgressBar key={d.name_kr} pct={d.probability}
                            color={d.probability > 0.5 ? C.red : d.probability > 0.2 ? C.amber : C.green}
                            label={d.name_kr} valueLabel={(d.probability * 100).toFixed(1) + "%"} />
                        ))}
                      </div>
                      {dis.abnormal_biomarkers && dis.abnormal_biomarkers.length > 0 && (
                        <div className="mt-3 text-textMuted text-[11px]">
                          이상 감지: {dis.abnormal_biomarkers.slice(0, 3).join(" / ")}
                        </div>
                      )}
                    </>
                  )}
                </ResultCard>

                {/* ── 부상 위험 ── */}
                <ResultCard title="부상 위험 예측" badge="부상 예측" accentColor={C.amber}>
                  {inj && (
                    <div className="flex gap-5 items-start">
                      <RiskGauge
                        score={inj.combined_risk_score}
                        label={inj.combined_risk_grade}
                        size={160}
                      />
                      <div className="flex-1 space-y-3 mt-1">
                        {inj.top3.slice(0, 3).map((injury) => (
                          <ProgressBar key={injury.name_kr} pct={injury.probability}
                            color={injury.probability > 0.4 ? C.red : C.amber}
                            label={injury.name_kr} valueLabel={(injury.probability * 100).toFixed(1) + "%"} />
                        ))}
                        <div className="text-textMuted text-[11px] mt-1">{inj.timeline}</div>
                      </div>
                    </div>
                  )}
                </ResultCard>

                {/* ── Chain-of-Reasoning ── */}
                <ResultCard title="Chain-of-Reasoning" badge="AI 추론" accentColor={C.purple}>
                  {rsn && (
                    <>
                      <div className="space-y-2 mb-3">
                        {rsn.reasoning_trace.map((step, i) => (
                          <div key={step.step} className="flex items-start gap-3">
                            <div className="flex flex-col items-center mt-1">
                              <div className="w-2 h-2 rounded-full"
                                style={{ background: i === rsn.reasoning_trace.length - 1 ? C.purple : "#32425B" }} />
                              {i < rsn.reasoning_trace.length - 1 && (
                                <div className="w-0.5 h-5 bg-border mt-0.5" />
                              )}
                            </div>
                            <div className="flex-1 flex justify-between text-[12px]">
                              <span className="text-textMuted">{step.label}</span>
                              <span className="font-semibold"
                                style={{ color: i === rsn.reasoning_trace.length - 1 ? C.purple : "#94A3B8" }}>
                                {step.prediction_kr}
                              </span>
                              <span className="text-textSec">
                                {(step.probability * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="text-textMuted text-[11px]">
                        불확실성: {(rsn.uncertainty * 100).toFixed(1)}% · 근거 강도: {(rsn.evidence_strength * 100).toFixed(1)}%
                        {rsn.is_demo_mode && " · 데모 모드"}
                      </div>
                    </>
                  )}
                </ResultCard>
              </div>

              {/* ── 개인 기준선 편차 ── */}
              {baseline && sample && (
                <section>
                  <h2 className="text-textSec text-[14px] font-semibold mb-3">
                    개인 기준선 편차
                    <span className="ml-2 text-textMuted text-[12px] font-normal">
                      최근 {baseline.sessions}회 기준
                    </span>
                  </h2>
                  <div className="bg-card rounded-xl p-5 grid grid-cols-2 gap-x-8 gap-y-3">
                    {Object.entries(sample.features).map(([key, val]) => {
                      const z = deviationZ(val, baseline, key);
                      const color = zColor(z, key);
                      const baseVal = baseline.mean[key];
                      return (
                        <div key={key} className="flex items-center justify-between border-b border-border/30 pb-2">
                          <span className="text-textSec text-[12px]">
                            {FEATURE_LABELS[key] ?? key}
                          </span>
                          <div className="flex items-center gap-3 text-right">
                            {baseVal != null && (
                              <span className="text-textMuted text-[11px]">
                                기준 {baseVal.toFixed(3)}
                              </span>
                            )}
                            <span className="text-textPri font-semibold text-[13px]">
                              {val.toFixed(3)}
                            </span>
                            {z != null && (
                              <span
                                className="text-[11px] font-semibold w-12 text-right"
                                style={{ color }}
                              >
                                {z > 0 ? "+" : ""}{z.toFixed(1)}σ
                              </span>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="mt-2 flex gap-4 text-[11px] text-textMuted">
                    <span><span style={{ color: C.green }}>●</span> 정상 범위 (&lt;1σ)</span>
                    <span><span style={{ color: C.amber }}>●</span> 주의 (1–2σ)</span>
                    <span><span style={{ color: C.red }}>●</span> 이탈 (&gt;2σ)</span>
                  </div>
                </section>
              )}
            </>
          )}
        </div>

        {/* 하단 바 */}
        <footer className="bg-surface border-t border-border px-8 py-2.5 text-textMuted text-[11px] flex justify-between shrink-0">
          <span>
            Shoealls Gait Analysis API · POST /api/v1/analyze ·{" "}
            {cls?.is_demo_mode ? "데모 모드" : "모델 로드됨"}
          </span>
          <span>© 2026 Shoealls</span>
        </footer>
      </main>
    </div>
  );
}
