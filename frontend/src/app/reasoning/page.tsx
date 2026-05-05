"use client";

import { useState, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import { ResultCard } from "@/components/ResultCard";
import { api, SampleResponse, ReasoningResponse } from "@/lib/api";

const PROFILES = ["normal", "parkinsons", "stroke", "fall_risk"] as const;
type Profile = (typeof PROFILES)[number];

const PROFILE_KR: Record<Profile, string> = {
  normal: "정상 보행",
  parkinsons: "파킨슨",
  stroke: "뇌졸중",
  fall_risk: "낙상 위험",
};

const C = { blue: "#3B82F6", purple: "#AF65FA", amber: "#F59E0B", red: "#EF4444", green: "#10B981" };

export default function ReasoningPage() {
  const [profile, setProfile] = useState<Profile>("parkinsons");
  const [_sample, setSample] = useState<SampleResponse | null>(null);
  const [result, setResult] = useState<ReasoningResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await api.sample(profile);
      setSample(s);
      const r = await api.analyze(s.sensor_data, s.features);
      setResult(r.reasoning);
    } catch (e) {
      setError(e instanceof Error ? e.message : "분석 실패");
    } finally {
      setLoading(false);
    }
  }, [profile]);

  const confidenceColor = (c: number) => c >= 0.7 ? C.green : c >= 0.4 ? C.amber : C.red;

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-textPri font-semibold text-xl">Chain-of-Reasoning AI 추론</h1>
            <div className="text-textMuted text-[12px] mt-0.5">멀티모달 데이터 → 단계별 추론 → 최종 진단</div>
          </div>
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
              className="bg-purple hover:bg-purple/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg transition-colors"
            >
              {loading ? "추론 중…" : "추론 실행"}
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {error && (
            <div className="bg-red/10 border border-red/30 text-red rounded-xl px-5 py-4 text-[14px]">
              오류: {error}
            </div>
          )}

          {loading && (
            <div className="bg-card rounded-2xl p-12 text-center">
              <div className="inline-block w-8 h-8 border-2 border-purple border-t-transparent rounded-full animate-spin mb-3" />
              <div className="text-textSec text-[14px]">Chain-of-Reasoning 추론 중…</div>
            </div>
          )}

          {result && (
            <>
              {/* 최종 결과 배너 */}
              <div
                className="rounded-xl p-5 flex items-center justify-between"
                style={{ background: `${C.purple}15`, border: `1px solid ${C.purple}40` }}
              >
                <div>
                  <div className="text-textMuted text-[12px] mb-1">최종 AI 판정</div>
                  <div className="text-textPri font-bold text-2xl">{result.final_prediction_kr}</div>
                </div>
                <div className="text-right">
                  <div className="text-purple font-bold text-3xl">{(result.confidence * 100).toFixed(1)}%</div>
                  <div className="text-textMuted text-[11px]">신뢰도</div>
                  {result.is_demo_mode && (
                    <span className="px-2 py-0.5 rounded-full text-[10px] bg-amber/20 text-amber mt-1 inline-block">
                      데모 모드
                    </span>
                  )}
                </div>
              </div>

              {/* 추론 단계 */}
              <ResultCard title="추론 단계 (Reasoning Trace)" badge="CoR" accentColor={C.purple}>
                <div className="space-y-0">
                  {result.reasoning_trace.map((step, i) => {
                    const isLast = i === result.reasoning_trace.length - 1;
                    return (
                      <div key={step.step} className="flex items-start gap-4">
                        <div className="flex flex-col items-center pt-1 w-6">
                          <div
                            className="w-3 h-3 rounded-full border-2 flex-shrink-0"
                            style={{
                              background: isLast ? C.purple : "transparent",
                              borderColor: isLast ? C.purple : "#32425B",
                            }}
                          />
                          {!isLast && <div className="w-0.5 bg-border flex-1 min-h-[28px] mt-0.5" />}
                        </div>
                        <div
                          className="flex-1 pb-4 flex items-start justify-between gap-4"
                          style={{ borderBottom: isLast ? "none" : "1px solid transparent" }}
                        >
                          <div>
                            <div className="text-[11px] text-textMuted uppercase tracking-wide mb-0.5">
                              Step {step.step}
                            </div>
                            <div className="text-textSec text-[13px]">{step.label}</div>
                          </div>
                          <div className="text-right flex-shrink-0">
                            <div
                              className="font-semibold text-[13px]"
                              style={{ color: isLast ? C.purple : "#94A3B8" }}
                            >
                              {step.prediction_kr}
                            </div>
                            <div
                              className="text-[12px] mt-0.5"
                              style={{ color: confidenceColor(step.probability) }}
                            >
                              {(step.probability * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div
                  className="mt-2 pt-3 border-t border-border text-[11px] flex gap-4"
                  style={{ color: "#64748B" }}
                >
                  <span>불확실성: {(result.uncertainty * 100).toFixed(1)}%</span>
                  <span>근거 강도: {(result.evidence_strength * 100).toFixed(1)}%</span>
                </div>
              </ResultCard>

              {/* 이상 발견 */}
              {result.anomaly_findings && result.anomaly_findings.length > 0 && (
                <ResultCard title="이상 감지 결과" badge="Anomaly" accentColor={C.amber}>
                  <div className="space-y-3">
                    {result.anomaly_findings.map((finding) => (
                      <div key={finding.modality}>
                        <div className="text-textSec text-[12px] font-semibold mb-1 uppercase tracking-wide">
                          {finding.modality}
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {finding.anomalies.map((a) => (
                            <span
                              key={a.type}
                              className="px-2 py-0.5 rounded text-[11px]"
                              style={{ color: C.amber, background: `${C.amber}20` }}
                            >
                              {a.type} ({Math.round(a.score * 100)}%)
                            </span>
                          ))}
                          {finding.anomalies.length === 0 && (
                            <span className="text-textMuted text-[11px]">이상 없음</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </ResultCard>
              )}

              {/* 한글 리포트 */}
              {result.report_kr && (
                <ResultCard title="AI 분석 리포트" badge="자연어 요약" accentColor={C.blue}>
                  <p className="text-textSec text-[13px] leading-relaxed whitespace-pre-line">
                    {result.report_kr}
                  </p>
                </ResultCard>
              )}
            </>
          )}

          {!result && !loading && (
            <div className="bg-card rounded-2xl p-12 text-center">
              <div className="text-textMuted text-[14px]">
                프로파일을 선택하고 <strong className="text-textSec">추론 실행</strong>을 눌러
                단계별 AI 추론 과정을 확인하세요.
              </div>
              <div className="mt-3 text-textMuted text-[12px]">
                센서 분석 → 이상 감지 → 패턴 분류 → 질환 예측 → 최종 판정
              </div>
            </div>
          )}
        </div>

        <footer className="bg-surface border-t border-border px-8 py-2.5 text-textMuted text-[11px] flex justify-between shrink-0">
          <span>POST /api/v1/analyze · reasoning 섹션 · Chain-of-Reasoning</span>
          <span>© 2026 Shoealls</span>
        </footer>
      </main>
    </div>
  );
}
