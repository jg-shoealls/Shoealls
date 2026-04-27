"use client";

import { useState, useCallback, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import ThemeToggle from "@/components/ThemeToggle";
import RiskGauge from "@/components/RiskGauge";
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
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]"); }
  catch { return []; }
}

function saveHistory(h: HistoryEntry[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(h.slice(0, MAX_HISTORY)));
}

// ── 상세 카드 (단일/비교 공통) ──────────────────────────────────────
function DetailCard({ entry }: { entry: HistoryEntry }) {
  const r = entry.result;
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h2 className="text-textPri font-semibold text-[15px]">
          {PROFILE_KR[entry.profile]} — {entry.timestamp}
        </h2>
        {r.classify.is_demo_mode && (
          <span className="px-2 py-0.5 rounded-full text-[10px] bg-amber/20 text-amber">데모 모드</span>
        )}
      </div>

      {/* 요약 4개 */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-surface rounded-xl p-3">
          <div className="text-textMuted text-[11px] mb-1">보행 분류</div>
          <div className="text-textPri font-bold text-[16px]">{r.classify.prediction_kr}</div>
          <div className="text-blue text-[12px]">{(r.classify.confidence * 100).toFixed(1)}% 신뢰도</div>
        </div>
        <div className="bg-surface rounded-xl p-3">
          <div className="text-textMuted text-[11px] mb-1">질환 ML 예측</div>
          <div className="text-textPri font-bold text-[16px]">{r.disease_risk.ml_prediction_kr}</div>
          <div className="text-green text-[12px]">{(r.disease_risk.ml_confidence * 100).toFixed(1)}% 신뢰도</div>
        </div>
      </div>

      {/* 부상 게이지 */}
      <div className="bg-surface rounded-xl p-4 flex items-center gap-4">
        <RiskGauge score={r.injury_risk.combined_risk_score} label={r.injury_risk.combined_risk_grade} size={140} />
        <div className="flex-1 space-y-2">
          <div className="text-textMuted text-[11px]">부상 Top 3</div>
          {r.injury_risk.top3.map((inj) => (
            <div key={inj.name_kr} className="flex items-center gap-2 text-[12px]">
              <span className="text-textSec w-20 truncate">{inj.name_kr}</span>
              <div className="flex-1 h-1.5 bg-card rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${inj.probability * 100}%`, background: inj.probability > 0.4 ? C.red : C.amber }}
                />
              </div>
              <span className="text-textPri font-semibold w-10 text-right">
                {(inj.probability * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* 질환 Top 3 */}
      <div className="bg-surface rounded-xl p-4">
        <div className="text-textSec text-[12px] font-semibold mb-3">질환 위험 Top 3</div>
        <div className="space-y-2">
          {r.disease_risk.ml_top3.map((d) => (
            <div key={d.name_kr} className="flex items-center justify-between text-[12px]">
              <span className="text-textSec">{d.name_kr}</span>
              <div className="flex items-center gap-2">
                <div className="w-24 h-1.5 bg-card rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${d.probability * 100}%`, background: riskColor(d.probability) }}
                  />
                </div>
                <span className="text-textPri font-semibold w-9 text-right">
                  {(d.probability * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* AI 리포트 */}
      {r.reasoning.report_kr && (
        <div className="bg-surface rounded-xl p-4">
          <div className="text-textSec text-[12px] font-semibold mb-2">AI 분석 리포트</div>
          <p className="text-textSec text-[12px] leading-relaxed whitespace-pre-line line-clamp-6">
            {r.reasoning.report_kr}
          </p>
        </div>
      )}
    </div>
  );
}

// ── 비교 수치 행 ─────────────────────────────────────────────────────
function CompareRow({
  label,
  v1,
  v2,
  color1,
  color2,
}: {
  label: string;
  v1: string;
  v2: string;
  color1?: string;
  color2?: string;
}) {
  return (
    <div className="grid grid-cols-3 text-[12px] border-b border-border/30 py-1.5">
      <span className="text-textMuted">{label}</span>
      <span className="font-semibold" style={{ color: color1 }}>{v1}</span>
      <span className="font-semibold" style={{ color: color2 }}>{v2}</span>
    </div>
  );
}

export default function HistoryPage() {
  const [profile, setProfile] = useState<Profile>("normal");
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<HistoryEntry | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [compareTarget, setCompareTarget] = useState<HistoryEntry | null>(null);

  useEffect(() => { setHistory(loadHistory()); }, []);

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
      setCompareTarget(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "분석 실패");
    } finally {
      setLoading(false);
    }
  }, [profile, history]);

  const clearHistory = () => {
    setHistory([]);
    setSelected(null);
    setCompareTarget(null);
    localStorage.removeItem(STORAGE_KEY);
  };

  const toggleCompareMode = () => {
    setCompareMode((m) => !m);
    setCompareTarget(null);
  };

  const handleSelect = (entry: HistoryEntry) => {
    if (!compareMode) {
      setSelected(entry);
      return;
    }
    if (!selected) { setSelected(entry); return; }
    if (entry.id === selected.id) return;
    setCompareTarget(entry);
  };

  const isSelectedForCompare = (id: string) =>
    id === selected?.id || id === compareTarget?.id;

  // ── 비교 요약 ───────────────────────────────────────────────────────
  const CompareView = () => {
    if (!selected || !compareTarget) return null;
    const a = selected;
    const b = compareTarget;
    return (
      <div className="space-y-5">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue/10 border border-blue/30 rounded-xl px-4 py-2 text-[12px] text-textSec">
            <span className="font-semibold text-blue">A</span> — {PROFILE_KR[a.profile]} · {a.timestamp}
          </div>
          <div className="bg-purple/10 border border-purple/30 rounded-xl px-4 py-2 text-[12px] text-textSec">
            <span className="font-semibold text-purple">B</span> — {PROFILE_KR[b.profile]} · {b.timestamp}
          </div>
        </div>

        <div className="bg-card rounded-xl p-4">
          <div className="grid grid-cols-3 text-[11px] font-semibold text-textMuted border-b border-border pb-2 mb-2">
            <span>항목</span>
            <span className="text-blue">A</span>
            <span className="text-purple">B</span>
          </div>
          <CompareRow
            label="보행 분류"
            v1={a.result.classify.prediction_kr}
            v2={b.result.classify.prediction_kr}
            color1={C.blue} color2={C.purple}
          />
          <CompareRow
            label="분류 신뢰도"
            v1={(a.result.classify.confidence * 100).toFixed(1) + "%"}
            v2={(b.result.classify.confidence * 100).toFixed(1) + "%"}
          />
          <CompareRow
            label="질환 예측"
            v1={a.result.disease_risk.ml_prediction_kr}
            v2={b.result.disease_risk.ml_prediction_kr}
            color1={C.blue} color2={C.purple}
          />
          <CompareRow
            label="질환 신뢰도"
            v1={(a.result.disease_risk.ml_confidence * 100).toFixed(1) + "%"}
            v2={(b.result.disease_risk.ml_confidence * 100).toFixed(1) + "%"}
          />
          <CompareRow
            label="부상 종합"
            v1={(a.result.injury_risk.combined_risk_score * 100).toFixed(1) + "%"}
            v2={(b.result.injury_risk.combined_risk_score * 100).toFixed(1) + "%"}
            color1={riskColor(a.result.injury_risk.combined_risk_score)}
            color2={riskColor(b.result.injury_risk.combined_risk_score)}
          />
          <CompareRow
            label="부상 등급"
            v1={a.result.injury_risk.combined_risk_grade}
            v2={b.result.injury_risk.combined_risk_grade}
          />
          <CompareRow
            label="AI 신뢰도"
            v1={(a.result.reasoning.confidence * 100).toFixed(1) + "%"}
            v2={(b.result.reasoning.confidence * 100).toFixed(1) + "%"}
          />
          <CompareRow
            label="불확실성"
            v1={(a.result.reasoning.uncertainty * 100).toFixed(1) + "%"}
            v2={(b.result.reasoning.uncertainty * 100).toFixed(1) + "%"}
          />
        </div>

        {/* 게이지 나란히 */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-card rounded-xl p-4 flex flex-col items-center">
            <div className="text-blue text-[11px] font-semibold mb-2">A 부상 위험</div>
            <RiskGauge score={a.result.injury_risk.combined_risk_score} label={a.result.injury_risk.combined_risk_grade} size={150} />
          </div>
          <div className="bg-card rounded-xl p-4 flex flex-col items-center">
            <div className="text-purple text-[11px] font-semibold mb-2">B 부상 위험</div>
            <RiskGauge score={b.result.injury_risk.combined_risk_score} label={b.result.injury_risk.combined_risk_grade} size={150} />
          </div>
        </div>
      </div>
    );
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
            {history.length >= 2 && (
              <button
                onClick={toggleCompareMode}
                className={`text-[13px] px-4 py-2 rounded-lg border transition-colors ${
                  compareMode
                    ? "bg-purple/15 border-purple/40 text-purple font-semibold"
                    : "border-border text-textSec hover:text-textPri"
                }`}
              >
                {compareMode ? "비교 모드 ON" : "비교 모드"}
              </button>
            )}
            <ThemeToggle />
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

        {compareMode && (
          <div className="bg-purple/10 border-b border-purple/20 px-8 py-2 text-[12px] text-purple shrink-0">
            목록에서 두 세션을 선택하세요 · A: {selected ? PROFILE_KR[selected.profile] + " " + selected.timestamp : "미선택"} · B: {compareTarget ? PROFILE_KR[compareTarget.profile] + " " + compareTarget.timestamp : "미선택"}
          </div>
        )}

        <div className="flex-1 flex overflow-hidden">
          {/* 이력 목록 */}
          <div className="w-72 shrink-0 border-r border-border overflow-y-auto p-4 space-y-2">
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
              const isA = selected?.id === entry.id;
              const isB = compareTarget?.id === entry.id;
              const isActive = isA || isB;
              return (
                <button
                  key={entry.id}
                  onClick={() => handleSelect(entry)}
                  className={`w-full text-left rounded-xl p-3.5 transition-colors border ${
                    isA
                      ? "bg-blue/10 border-blue/40"
                      : isB
                      ? "bg-purple/10 border-purple/40"
                      : "bg-card border-border hover:border-blue/30"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      {isA && <span className="text-blue text-[10px] font-bold">A</span>}
                      {isB && <span className="text-purple text-[10px] font-bold">B</span>}
                      <span className="text-textPri font-semibold text-[13px]">
                        {PROFILE_KR[entry.profile]}
                      </span>
                    </div>
                    <span className="text-[11px] font-semibold" style={{ color: riskColor(risk) }}>
                      {(risk * 100).toFixed(0)}% 위험
                    </span>
                  </div>
                  <div className="text-textSec text-[12px]">{entry.result.classify.prediction_kr}</div>
                  <div className="text-textMuted text-[10px] mt-1">{entry.timestamp}</div>
                </button>
              );
            })}
          </div>

          {/* 상세 / 비교 패널 */}
          <div className="flex-1 overflow-y-auto p-6">
            {compareMode && selected && compareTarget ? (
              <CompareView />
            ) : selected ? (
              <DetailCard entry={selected} />
            ) : (
              <div className="flex items-center justify-center h-full text-textMuted text-[14px]">
                {compareMode
                  ? "두 세션을 선택하면 비교 분석이 표시됩니다."
                  : "왼쪽 목록에서 분석 기록을 선택하세요."}
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
