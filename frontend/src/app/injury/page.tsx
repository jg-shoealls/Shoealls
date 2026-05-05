"use client";

import { useState, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import { ResultCard, ProgressBar } from "@/components/ResultCard";
import { api, SampleResponse, InjuryRiskResponse } from "@/lib/api";

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
};

const INJURY_ICONS: Record<string, string> = {
  "족저근막염":       "🦶",
  "중족골 피로골절":  "🦴",
  "발목 염좌":        "🔄",
  "종골 스트레스":    "⚡",
  "평발/과회내":      "⬇",
  "요족/과회외":      "⬆",
};

export default function InjuryPage() {
  const [profile, setProfile] = useState<Profile>("fall_risk");
  const [_sample, setSample] = useState<SampleResponse | null>(null);
  const [result, setResult] = useState<InjuryRiskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await api.sample(profile);
      setSample(s);
      const r = await api.analyze(s.sensor_data, s.features);
      setResult(r.injury_risk);
    } catch (e) {
      setError(e instanceof Error ? e.message : "분석 실패");
    } finally {
      setLoading(false);
    }
  }, [profile]);

  const gradeColor = (grade: string) => {
    if (grade.includes("위험")) return C.red;
    if (grade.includes("경고")) return C.amber;
    if (grade.includes("주의")) return "#FBBF24";
    return C.green;
  };

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <h1 className="text-textPri font-semibold text-xl">부상 위험 예측</h1>
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
              {loading ? "분석 중…" : "위험 평가"}
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
              <div className="inline-block w-8 h-8 border-2 border-blue border-t-transparent rounded-full animate-spin mb-3" />
              <div className="text-textSec text-[14px]">부상 위험도 평가 중…</div>
            </div>
          )}

          {result && (
            <>
              {/* 종합 위험도 헤더 */}
              <div className="bg-card rounded-xl p-6 flex items-center gap-8">
                <div className="text-center">
                  <div
                    className="text-5xl font-bold"
                    style={{ color: gradeColor(result.combined_risk_grade) }}
                  >
                    {(result.combined_risk_score * 100).toFixed(0)}%
                  </div>
                  <div className="text-textSec text-[12px] mt-1">종합 위험도</div>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span
                      className="px-3 py-1 rounded-full text-[13px] font-semibold"
                      style={{ color: gradeColor(result.combined_risk_grade), background: `${gradeColor(result.combined_risk_grade)}20` }}
                    >
                      {result.combined_risk_grade}
                    </span>
                    <span className="text-textSec text-[13px]">
                      주요 위험: <strong className="text-textPri">{result.predicted_injury_kr}</strong>
                    </span>
                  </div>
                  <ProgressBar
                    pct={result.combined_risk_score}
                    color={gradeColor(result.combined_risk_grade)}
                    label=""
                    valueLabel=""
                  />
                  <div className="text-textMuted text-[11px] mt-2">{result.timeline}</div>
                </div>
              </div>

              {/* 부상 유형별 상세 */}
              <ResultCard title="부상 유형별 위험도" badge="6-type 분석" accentColor={C.amber}>
                <div className="grid grid-cols-2 gap-4">
                  {result.top3.map((inj) => (
                    <div key={inj.name_kr} className="bg-surface rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-[18px]">{INJURY_ICONS[inj.name_kr] ?? "⚠"}</span>
                        <span className="text-textPri font-semibold text-[13px]">{inj.name_kr}</span>
                      </div>
                      <ProgressBar
                        pct={inj.probability}
                        color={inj.probability > 0.4 ? C.red : C.amber}
                        label=""
                        valueLabel={(inj.probability * 100).toFixed(1) + "%"}
                      />
                    </div>
                  ))}
                </div>
              </ResultCard>

              {/* 우선 조치 */}
              {result.priority_actions && result.priority_actions.length > 0 && (
                <ResultCard title="권장 조치 사항" badge="액션 플랜" accentColor={C.blue}>
                  <ul className="space-y-2">
                    {result.priority_actions.map((action, i) => (
                      <li key={i} className="flex items-start gap-3 text-[13px] text-textSec">
                        <span className="text-blue font-bold mt-0.5">{i + 1}.</span>
                        {action}
                      </li>
                    ))}
                  </ul>
                </ResultCard>
              )}
            </>
          )}

          {!result && !loading && (
            <div className="bg-card rounded-2xl p-12 text-center">
              <div className="text-textMuted text-[14px]">
                프로파일을 선택하고 <strong className="text-textSec">위험 평가</strong>를 눌러 부상 위험을 예측하세요.
              </div>
              <div className="mt-3 text-textMuted text-[12px]">6가지 부상 유형 · 족저압 기반 바이오역학 분석</div>
            </div>
          )}
        </div>

        <footer className="bg-surface border-t border-border px-8 py-2.5 text-textMuted text-[11px] flex justify-between shrink-0">
          <span>POST /api/v1/analyze · injury_risk 섹션</span>
          <span>© 2026 Shoealls</span>
        </footer>
      </main>
    </div>
  );
}
