"use client";

import { useState, useCallback, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import { api, AnalyzeResponse } from "@/lib/api";

const PROFILES = ["normal", "parkinsons", "stroke", "fall_risk"] as const;
type Profile = (typeof PROFILES)[number];

const PROFILE_KR: Record<Profile, string> = {
  normal: "정상 보행",
  parkinsons: "파킨슨",
  stroke: "뇌졸중",
  fall_risk: "낙상 위험",
};

interface HistoryEntry {
  id: string;
  timestamp: string;
  profile: Profile;
  result: AnalyzeResponse;
}

const STORAGE_KEY = "shoealls_history";
const MAX_HISTORY = 20;

const C = { blue: "#3B82F6", green: "#10B981", amber: "#F59E0B", red: "#EF4444", purple: "#AF65FA" };

function riskColor(score: number) {
  if (score > 0.65) return C.red;
  if (score > 0.35) return C.amber;
  return C.green;
}

function loadHistory(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
  } catch {
    return [];
  }
}

function saveHistory(h: HistoryEntry[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(h.slice(0, MAX_HISTORY)));
}

export default function HistoryPage() {
  const [profile, setProfile] = useState<Profile>("normal");
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<HistoryEntry | null>(null);

  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  const runAndSave = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await api.sample(profile);
      const r = await api.analyze(s.sensor_data, s.features);
      const entry: HistoryEntry = {
        id: Math.random().toString(36).slice(2, 10),
        timestamp: new Date().toLocaleString("ko-KR"),
        profile,
        result: r,
      };
      const updated = [entry, ...history];
      setHistory(updated);
      saveHistory(updated);
      setSelected(entry);
    } catch (e) {
      setError(e instanceof Error ? e.message : "분석 실패");
    } finally {
      setLoading(false);
    }
  }, [profile, history]);

  const clearHistory = () => {
    setHistory([]);
    setSelected(null);
    localStorage.removeItem(STORAGE_KEY);
  };

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-textPri font-semibold text-xl">분석 이력</h1>
            <div className="text-textMuted text-[12px] mt-0.5">최근 {MAX_HISTORY}건 · 브라우저 로컬 저장</div>
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
              onClick={runAndSave}
              disabled={loading}
              className="bg-blue hover:bg-blue/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg transition-colors"
            >
              {loading ? "분석 중…" : "분석 후 저장"}
            </button>
            {history.length > 0 && (
              <button
                onClick={clearHistory}
                className="text-textMuted hover:text-red text-[13px] px-3 py-2 rounded-lg border border-border hover:border-red/40 transition-colors"
              >
                이력 삭제
              </button>
            )}
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden">
          {/* 이력 목록 */}
          <div className="w-80 shrink-0 border-r border-border overflow-y-auto p-4 space-y-2">
            {error && (
              <div className="bg-red/10 border border-red/30 text-red rounded-xl px-4 py-3 text-[12px]">
                {error}
              </div>
            )}
            {history.length === 0 && !loading && (
              <div className="text-center py-12 text-textMuted text-[13px]">
                분석 결과가 없습니다.<br />분석 후 저장을 눌러보세요.
              </div>
            )}
            {history.map((entry) => {
              const risk = entry.result.injury_risk.combined_risk_score;
              const isSelected = selected?.id === entry.id;
              return (
                <button
                  key={entry.id}
                  onClick={() => setSelected(entry)}
                  className={`w-full text-left rounded-xl p-3.5 transition-colors border ${
                    isSelected
                      ? "bg-blue/10 border-blue/40"
                      : "bg-card border-border hover:border-blue/30"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-textPri font-semibold text-[13px]">
                      {PROFILE_KR[entry.profile]}
                    </span>
                    <span
                      className="text-[11px] font-semibold"
                      style={{ color: riskColor(risk) }}
                    >
                      {(risk * 100).toFixed(0)}% 위험
                    </span>
                  </div>
                  <div className="text-textSec text-[12px]">{entry.result.classify.prediction_kr}</div>
                  <div className="text-textMuted text-[10px] mt-1">{entry.timestamp}</div>
                </button>
              );
            })}
          </div>

          {/* 선택 상세 */}
          <div className="flex-1 overflow-y-auto p-6">
            {selected ? (
              <div className="space-y-5">
                <div className="flex items-center gap-3 mb-2">
                  <h2 className="text-textPri font-semibold text-[16px]">
                    {PROFILE_KR[selected.profile]} — {selected.timestamp}
                  </h2>
                  {selected.result.classify.is_demo_mode && (
                    <span className="px-2 py-0.5 rounded-full text-[10px] bg-amber/20 text-amber">데모 모드</span>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-card rounded-xl p-4">
                    <div className="text-textMuted text-[11px] mb-1">보행 분류</div>
                    <div className="text-textPri font-bold text-[17px]">
                      {selected.result.classify.prediction_kr}
                    </div>
                    <div className="text-blue text-[13px]">
                      {(selected.result.classify.confidence * 100).toFixed(1)}% 신뢰도
                    </div>
                  </div>
                  <div className="bg-card rounded-xl p-4">
                    <div className="text-textMuted text-[11px] mb-1">질환 ML 예측</div>
                    <div className="text-textPri font-bold text-[17px]">
                      {selected.result.disease_risk.ml_prediction_kr}
                    </div>
                    <div className="text-green text-[13px]">
                      {(selected.result.disease_risk.ml_confidence * 100).toFixed(1)}% 신뢰도
                    </div>
                  </div>
                  <div className="bg-card rounded-xl p-4">
                    <div className="text-textMuted text-[11px] mb-1">부상 종합 위험도</div>
                    <div
                      className="font-bold text-[17px]"
                      style={{ color: riskColor(selected.result.injury_risk.combined_risk_score) }}
                    >
                      {(selected.result.injury_risk.combined_risk_score * 100).toFixed(1)}%
                    </div>
                    <div className="text-textSec text-[13px]">{selected.result.injury_risk.combined_risk_grade}</div>
                  </div>
                  <div className="bg-card rounded-xl p-4">
                    <div className="text-textMuted text-[11px] mb-1">AI 추론 신뢰도</div>
                    <div
                      className="font-bold text-[17px]"
                      style={{ color: C.purple }}
                    >
                      {(selected.result.reasoning.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-textSec text-[13px]">
                      불확실성 {(selected.result.reasoning.uncertainty * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* 질환 top3 */}
                <div className="bg-card rounded-xl p-4">
                  <div className="text-textSec text-[12px] font-semibold mb-3">질환 위험 Top 3</div>
                  <div className="space-y-2">
                    {selected.result.disease_risk.ml_top3.map((d) => (
                      <div key={d.name_kr} className="flex items-center justify-between text-[13px]">
                        <span className="text-textSec">{d.name_kr}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-32 h-1.5 bg-surface rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${d.probability * 100}%`,
                                background: riskColor(d.probability),
                              }}
                            />
                          </div>
                          <span className="text-textPri font-semibold w-10 text-right">
                            {(d.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 리포트 */}
                {selected.result.reasoning.report_kr && (
                  <div className="bg-card rounded-xl p-4">
                    <div className="text-textSec text-[12px] font-semibold mb-2">AI 분석 리포트</div>
                    <p className="text-textSec text-[13px] leading-relaxed whitespace-pre-line">
                      {selected.result.reasoning.report_kr}
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-textMuted text-[14px]">
                왼쪽 목록에서 분석 기록을 선택하세요.
              </div>
            )}
          </div>
        </div>

        <footer className="bg-surface border-t border-border px-8 py-2.5 text-textMuted text-[11px] flex justify-between shrink-0">
          <span>분석 이력 · localStorage 저장 · 최대 {MAX_HISTORY}건</span>
          <span>© 2026 Shoealls</span>
        </footer>
      </main>
    </div>
  );
}
