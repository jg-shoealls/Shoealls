"use client";

import { useState, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import { ResultCard, ProgressBar } from "@/components/ResultCard";
import { api, AnalyzeResponse, SampleResponse } from "@/lib/api";

const PROFILES = ["normal", "parkinsons", "stroke", "fall_risk"] as const;
type Profile = (typeof PROFILES)[number];

const PROFILE_KR: Record<Profile, string> = {
  normal: "정상 보행",
  parkinsons: "파킨슨",
  stroke: "뇌졸중",
  fall_risk: "낙상 위험",
};

const C = {
  blue:   "#2563EB",  // primaryBlue
  green:  "#0D9488",  // medicalTeal
  amber:  "#D97706",  // warningAmber
  red:    "#DC2626",  // dangerRed
  purple: "#4F46E5",  // accentIndigo
};

const CLASS_KR: Record<string, string> = {
  normal:       "정상 보행",
  antalgic:     "절뚝거림",
  ataxic:       "운동실조",
  parkinsonian: "파킨슨",
};

export default function Dashboard() {
  const [profile, setProfile] = useState<Profile>("parkinsons");
  const [sample, setSample] = useState<SampleResponse | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await api.sample(profile);
      setSample(s);
      const r = await api.analyze(s.sensor_data, s.features);
      setResult(r);
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
    <div className="flex h-screen bg-bg overflow-hidden relative">
      {/* 백그라운드 메쉬 그라디언트 */}
      <div className="absolute inset-0 bg-mesh opacity-80 pointer-events-none z-0" />

      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden relative z-10">
        {/* 탑바 (Glassmorphism) */}
        <header className="glass border-b border-border px-10 py-5 flex items-center justify-between shrink-0 shadow-sm relative z-20">
          <h1 className="text-textPri font-bold text-4xl tracking-tight flex items-center gap-3">
            보행 분석 대시보드
            <span className="px-3 py-1 rounded-full text-base font-bold tracking-wider text-primaryBlue bg-primaryBlue/10 border border-primaryBlue/20">
              PRO
            </span>
          </h1>
          <div className="flex items-center gap-4">
            <div className="relative group">
              <select
                value={profile}
                onChange={(e) => setProfile(e.target.value as Profile)}
                className="appearance-none bg-surface border border-border text-textPri text-lg font-medium rounded-xl px-5 py-4 pr-10 focus:outline-none focus:border-primaryBlue focus:ring-1 focus:ring-primaryBlue transition-all cursor-pointer shadow-sm backdrop-blur-md"
              >
                {PROFILES.map((p) => (
                  <option key={p} value={p} className="bg-bg text-textPri">{PROFILE_KR[p]}</option>
                ))}
              </select>
              <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-textSec group-hover:text-neonBlue transition-colors">
                ▼
              </div>
            </div>
            
            {cls?.is_demo_mode !== false && (
              <span className="px-3 py-1.5 rounded-xl text-lg font-bold text-amber bg-amber/10 border border-amber/20 shadow-[0_0_10px_rgba(245,158,11,0.2)]">
                데모 모드
              </span>
            )}
            
            <button
              onClick={runAnalysis}
              disabled={loading}
              className="relative overflow-hidden bg-primaryBlue hover:bg-primaryBlue/90 disabled:opacity-50 text-white font-bold text-xl px-12 py-4 rounded-xl transition-all shadow-md hover:-translate-y-0.5 active:translate-y-0 disabled:transform-none disabled:shadow-none disabled:cursor-not-allowed"
            >
              {loading ? (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  분석 중…
                </div>
              ) : "분석 시작"}
            </button>
          </div>
        </header>

        {/* 스크롤 영역 */}
        <div className="flex-1 overflow-y-auto p-10 space-y-8 relative z-10">
          {error && (
            <div className="glass border border-red/40 text-red rounded-2xl px-6 py-5 text-xl font-medium shadow-[0_0_20px_rgba(239,68,68,0.2)] animate-fade-in">
              <div className="flex items-center gap-3">
                <span className="text-xl">⚠️</span>
                오류: {error}
              </div>
            </div>
          )}

          {/* 센서 입력 요약 */}
          <section className="animate-fade-in-up">
            <h2 className="text-textPri text-2xl font-bold tracking-wide mb-4 flex items-center gap-2">
              <span className="w-1.5 h-6 bg-neonBlue rounded-full shadow-[0_0_10px_rgba(0,229,255,0.8)]" />
              센서 데이터 수집
            </h2>
            <div className="grid grid-cols-3 gap-6">
              {[
                { title: "IMU 센서", sub: "가속도 + 자이로스코프 6ch", val: "128 프레임", color: C.blue },
                { title: "족저압 센서", sub: "16 × 8 고해상도 그리드", val: sample ? "데이터 수신됨" : "대기 중", color: C.green },
                { title: "스켈레톤", sub: "17 관절 × 3D 모션 트래킹", val: "128 프레임", color: C.purple },
              ].map((card) => (
                <div key={card.title} className="glass-card rounded-[20px] p-6 flex flex-col gap-1.5 group relative overflow-hidden">
                  <div className="absolute right-0 top-0 w-24 h-24 rounded-bl-full opacity-10 pointer-events-none transition-transform duration-500 group-hover:scale-150" style={{ background: card.color }} />
                  
                  <div className="flex items-center gap-3 relative z-10">
                    <span className="w-3 h-3 rounded-full relative">
                      <span className="absolute inset-0 rounded-full animate-ping opacity-60" style={{ background: card.color }} />
                      <span className="absolute inset-0 rounded-full" style={{ background: card.color, boxShadow: `0 0 10px ${card.color}` }} />
                    </span>
                    <span className="text-textPri font-bold text-2xl tracking-wide">{card.title}</span>
                  </div>
                  <span className="text-textSec text-lg relative z-10 mt-1">{card.sub}</span>
                  <span className="text-xl font-bold mt-2 relative z-10" style={{ color: card.color, textShadow: `0 0 10px ${card.color}40` }}>
                    {card.val}
                  </span>
                </div>
              ))}
            </div>
          </section>

          {/* 초기 안내 */}
          {!result && !loading && (
            <div className="glass-card border-dashed border-2 border-white/10 rounded-[30px] p-16 text-center animate-fade-in flex flex-col items-center justify-center min-h-[300px]">
              <div className="w-20 h-20 rounded-full bg-gradient-to-br from-white/5 to-white/10 border border-white/10 flex items-center justify-center mb-6 shadow-inner">
                <span className="text-4xl text-neonBlue">⚡</span>
              </div>
              <div className="text-textPri text-3xl font-medium tracking-wide mb-2">
                보행 데이터를 분석할 준비가 되었습니다
              </div>
              <div className="text-textSec text-xl">
                우측 상단에서 프로파일을 선택하고 <strong className="text-neonBlue font-semibold">분석 시작</strong>을 클릭하세요
              </div>
            </div>
          )}

          {/* 로딩 */}
          {loading && (
            <div className="glass-card rounded-[30px] p-16 text-center flex flex-col items-center justify-center min-h-[300px] animate-pulse">
              <div className="relative w-20 h-20 mb-6">
                <div className="absolute inset-0 border-4 border-neonBlue/20 rounded-full" />
                <div className="absolute inset-0 border-4 border-neonBlue border-t-transparent rounded-full animate-spin shadow-[0_0_15px_rgba(0,229,255,0.5)]" />
                <div className="absolute inset-0 flex items-center justify-center text-neonBlue font-bold text-xs">AI</div>
              </div>
              <div className="text-textPri text-2xl font-medium tracking-widest uppercase mb-1">
                Deep Analysis Running
              </div>
              <div className="text-textSec text-lg">
                멀티모달 AI 엔진이 데이터를 추론하고 있습니다...
              </div>
            </div>
          )}

          {/* 결과 카드 그리드 */}
          {result && (
            <section className="animate-fade-in">
              <h2 className="text-textPri text-[16px] font-bold tracking-wide mb-5 flex items-center gap-2">
                <span className="w-1.5 h-6 bg-purple rounded-full shadow-[0_0_10px_rgba(217,70,239,0.8)]" />
                AI 분석 결과 레포트
              </h2>
              <div className="grid grid-cols-2 gap-6">

                {/* ── 보행 패턴 분류 ── */}
                <ResultCard title="보행 패턴 분류" badge="Pattern Class" accentColor={C.blue} isDemo={cls?.is_demo_mode} delay="0s">
                  <div className="flex items-center gap-8 h-full">
                    <div className="w-36 h-36 shrink-0 rounded-full bg-black/40 flex flex-col items-center justify-center border-4 border-blue/40 relative shadow-[inset_0_0_20px_rgba(0,229,255,0.2),_0_0_30px_rgba(0,229,255,0.15)]">
                      <svg className="absolute inset-0 w-full h-full -rotate-90">
                        <circle cx="70" cy="70" r="66" fill="transparent" stroke="rgba(255,255,255,0.05)" strokeWidth="4" />
                        <circle 
                          cx="70" cy="70" r="66" fill="transparent" 
                          stroke={C.blue} strokeWidth="4" 
                          strokeDasharray={414} 
                          strokeDashoffset={414 - (414 * (cls?.confidence ?? 0))}
                          className="transition-all duration-1000 ease-out drop-shadow-[0_0_8px_rgba(0,229,255,0.8)]"
                        />
                      </svg>
                      <span className="text-blue text-xl font-bold tracking-wider uppercase mb-1 drop-shadow-md">{cls?.prediction_kr}</span>
                      <span className="text-textPri font-black text-5xl drop-shadow-[0_2px_4px_rgba(0,0,0,0.1)] dark:drop-shadow-[0_2px_4px_rgba(0,0,0,0.5)]">
                        {cls ? (cls.confidence * 100).toFixed(1) : "--"}
                        <span className="text-2xl text-textSec ml-0.5">%</span>
                      </span>
                    </div>
                    <div className="flex-1 flex flex-col justify-center space-y-4">
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
                <ResultCard title="질환 예측 엔진" badge="Disease Risk" accentColor={C.green} delay="0.1s">
                  {dis && (
                    <div className="flex flex-col h-full justify-between">
                      <div className="bg-black/20 rounded-xl p-4 border border-green/20 flex items-center justify-between mb-5 shadow-inner">
                        <div>
                          <div className="text-textSec text-lg uppercase tracking-wider mb-1">Primary Prediction</div>
                          <div className="text-textPri font-bold text-2xl">{dis.ml_prediction_kr}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-green font-black text-4xl drop-shadow-[0_0_10px_rgba(5,213,158,0.5)]">
                            {(dis.ml_confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-4 flex-1">
                        {dis.ml_top3.map((d) => (
                          <ProgressBar key={d.name_kr} pct={d.probability}
                            color={d.probability > 0.5 ? C.red : d.probability > 0.2 ? C.amber : C.green}
                            label={d.name_kr} valueLabel={(d.probability * 100).toFixed(1) + "%"} />
                        ))}
                      </div>
                      
                      {dis.abnormal_biomarkers && dis.abnormal_biomarkers.length > 0 && (
                        <div className="mt-5 pt-4 border-t border-white/5">
                          <div className="text-textSec text-base uppercase tracking-wider mb-2">이상 징후 감지</div>
                          <div className="flex flex-wrap gap-2">
                            {dis.abnormal_biomarkers.slice(0, 3).map(bm => (
                              <span key={bm} className="px-2.5 py-1 bg-black/5 dark:bg-white/5 border border-border rounded-lg text-textPri text-lg">
                                {bm}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </ResultCard>

                {/* ── 부상 위험 ── */}
                <ResultCard title="역학적 부상 위험" badge="Injury Warning" accentColor={C.amber} delay="0.2s">
                  {inj && (
                    <div className="flex gap-8 h-full items-center">
                      <div className="shrink-0 text-center">
                        <div className="relative inline-flex items-center justify-center w-32 h-32">
                          <div className="absolute inset-0 rounded-full border-4 border-amber/20" />
                          <div className={`absolute inset-0 rounded-full border-4 border-l-transparent animate-spin-slow opacity-80 ${inj.combined_risk_score > 0.7 ? 'border-red shadow-[0_0_20px_rgba(239,68,68,0.5)]' : 'border-amber'}`} style={{ animationDuration: '3s' }} />
                          {inj.combined_risk_score > 0.7 && (
                            <div className="absolute inset-0 rounded-full animate-ping opacity-20 bg-red" />
                          )}
                          <div className="flex flex-col items-center">
                            <span className={`font-black text-5xl drop-shadow-[0_0_15px_rgba(245,158,11,0.6)] ${inj.combined_risk_score > 0.7 ? 'text-red' : 'text-amber'}`}>
                              {(inj.combined_risk_score * 100).toFixed(0)}<span className="text-2xl">%</span>
                            </span>
                            <span className="text-textSec text-base uppercase tracking-wider mt-1">Risk Score</span>
                          </div>
                        </div>
                        <div className="mt-4">
                          <span className="px-4 py-1.5 rounded-xl text-lg font-bold tracking-widest uppercase"
                            style={inj.combined_risk_score > 0.7 
                              ? { color: C.red, background: `${C.red}20`, border: `1px solid ${C.red}40`, boxShadow: `0 0 15px ${C.red}30` }
                              : { color: C.amber, background: `${C.amber}20`, border: `1px solid ${C.amber}40` }}>
                            {inj.combined_risk_grade}
                          </span>
                        </div>
                      </div>
                      
                      <div className="flex-1 flex flex-col justify-between space-y-3 h-full py-1">
                        <div>
                          <div className="text-textSec text-base uppercase tracking-wider mb-2">Primary Risk Factors</div>
                          <div className="space-y-2">
                            {inj.top3.slice(0, 2).map((injury) => (
                              <ProgressBar key={injury.name_kr} pct={injury.probability}
                                color={injury.probability > 0.4 ? C.red : C.amber}
                                label={injury.name_kr} valueLabel={(injury.probability * 100).toFixed(1) + "%"} />
                            ))}
                          </div>
                        </div>
                        
                        <div className="bg-red/5 border border-red/20 rounded-xl p-3 flex-1 flex flex-col justify-center shadow-inner relative overflow-hidden">
                          <div className="absolute top-0 right-0 w-16 h-16 bg-red/10 rounded-bl-full pointer-events-none" />
                          <div className="flex items-center gap-2 mb-2 relative z-10">
                            <span className="w-2 h-2 rounded-full bg-red animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)]" />
                            <span className="text-red font-bold text-lg tracking-wide">AI 추천 우선 행동</span>
                          </div>
                          <ul className="space-y-1.5 relative z-10">
                            {inj.priority_actions && inj.priority_actions.slice(0, 3).map((action, idx) => (
                              <li key={idx} className="text-textPri text-lg flex items-start gap-2 leading-tight">
                                <span className="text-red mt-0.5 text-[10px]">▶</span>
                                {action}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  )}
                </ResultCard>

                {/* ── Chain-of-Reasoning ── */}
                <ResultCard title="인공지능 추론 과정" badge="Chain-of-Reasoning" accentColor={C.purple} delay="0.3s">
                  {rsn && (
                    <div className="flex flex-col h-full justify-between">
                      <div className="space-y-4 mb-4 flex-1">
                        {rsn.reasoning_trace.map((step, i) => {
                          const isLast = i === rsn.reasoning_trace.length - 1;
                          const color = isLast ? C.purple : "#64748B";
                          return (
                            <div key={step.step} className="flex items-start gap-4 relative">
                              {i < rsn.reasoning_trace.length - 1 && (
                                <div className="absolute left-1.5 top-5 w-0.5 h-[calc(100%+4px)] bg-border" />
                              )}
                              <div className="flex flex-col items-center mt-1 z-10">
                                <div className="w-3.5 h-3.5 rounded-full border-2 border-bg"
                                  style={{ background: color, boxShadow: isLast ? `0 0 10px ${color}` : 'none' }} />
                              </div>
                              <div className={`flex-1 flex justify-between items-center p-3 rounded-xl border ${isLast ? 'bg-purple/10 border-purple/30' : 'bg-black/5 dark:bg-white/5 border-transparent'} transition-colors`}>
                                <span className="text-textSec text-base w-24">{step.label}</span>
                                <span className={`font-bold text-xl ${isLast ? '' : 'text-textPri'}`}
                                  style={isLast ? { color: C.purple } : {}}>
                                  {step.prediction_kr}
                                </span>
                                <span className={`text-lg font-medium ${isLast ? 'text-textPri' : 'text-textMuted'}`}>
                                  {(step.probability * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      
                      <div className="flex items-center gap-4 p-3 bg-black/5 dark:bg-black/30 rounded-xl border border-border">
                        <div className="flex-1">
                          <div className="text-textMuted text-sm uppercase tracking-wider mb-1">Uncertainty</div>
                          <div className="text-textPri font-medium text-lg">{(rsn.uncertainty * 100).toFixed(1)}%</div>
                        </div>
                        <div className="w-px h-6 bg-border" />
                        <div className="flex-1">
                          <div className="text-textMuted text-sm uppercase tracking-wider mb-1">Evidence</div>
                          <div className="text-textPri font-medium text-lg">{(rsn.evidence_strength * 100).toFixed(1)}%</div>
                        </div>
                        {rsn.is_demo_mode && (
                          <>
                            <div className="w-px h-6 bg-border" />
                            <div className="flex-1 text-center text-amber text-sm font-bold tracking-wider">DEMO</div>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </ResultCard>
              </div>

              {/* ── AI 종합 소견서 (Full Width) ── */}
              {rsn?.report_kr && (
                <div className="mt-6 glass-card rounded-[24px] p-8 border border-white/10 relative overflow-hidden group animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
                  <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-gradient-to-b from-blue to-purple shadow-[0_0_15px_rgba(59,130,246,0.6)]" />
                  <div className="absolute right-0 bottom-0 w-64 h-64 bg-blue/10 rounded-full blur-3xl pointer-events-none group-hover:bg-blue/20 transition-all duration-700" />
                  
                  <div className="flex items-center gap-3 mb-4 relative z-10">
                    <span className="text-2xl drop-shadow-[0_0_10px_rgba(255,255,255,0.3)]">🤖</span>
                    <h3 className="text-textPri font-bold text-3xl tracking-wide">AI 멀티모달 종합 소견서</h3>
                    <span className="px-2.5 py-1 rounded-full text-base font-bold tracking-wider text-purple bg-purple/10 border border-purple/30 ml-2 shadow-[0_0_10px_rgba(217,70,239,0.2)]">
                      Llama 3 Generated
                    </span>
                  </div>
                  
                  <div className="relative z-10 bg-black/20 dark:bg-white/5 rounded-xl p-6 border border-border">
                    <p className="text-textPri text-2xl leading-relaxed font-medium">
                      {rsn.report_kr}
                    </p>
                  </div>
                </div>
              )}
            </section>
          )}
        </div>

        {/* 하단 바 (Glassmorphism) */}
        <footer className="glass border-t border-white/5 px-8 py-3 text-textMuted text-base flex justify-between shrink-0 relative z-20">
          <span className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-neonBlue animate-pulse" />
            Shoealls Advanced Gait AI · POST /api/v1/analyze
          </span>
          <span className="tracking-wide">© 2026 Shoealls. All rights reserved.</span>
        </footer>
      </main>
    </div>
  );
}
