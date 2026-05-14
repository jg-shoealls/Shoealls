"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import Sidebar from "@/components/Sidebar";
import { buildGaitReport } from "@/lib/gaitAnalysis";
import { generateMockSensorData, PROFILE_LABELS, type MockProfile } from "@/lib/mockSensorData";
import type { GaitFeatures, GaitReport } from "@/types/sensor";

const PROFILES: MockProfile[] = ["normal", "parkinsons", "stroke", "fall_risk"];
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

const RISK_COLORS: Record<string, string> = {
  low: "#10b981", watch: "#3b82f6", elevated: "#f59e0b", high: "#ef4444",
};
const RISK_KR: Record<string, string> = {
  low: "낮음", watch: "관찰", elevated: "주의", high: "높음",
};
const RISK_BG: Record<string, string> = {
  low: "rgba(16,185,129,0.07)", watch: "rgba(59,130,246,0.07)",
  elevated: "rgba(245,158,11,0.07)", high: "rgba(239,68,68,0.07)",
};

interface Snapshot {
  date: string; profile: MockProfile; label: string;
  score: number; symmetry: number; rhythm: number; stability: number; weight: number;
  speed: number; cadence: number; risk: string; riskScore: number;
  patterns: string[]; recommendation: string;
}

function buildSnapshot(profile: MockProfile, report: GaitReport, features: GaitFeatures): Snapshot {
  return {
    date: new Date().toLocaleDateString("ko-KR", { year: "numeric", month: "2-digit", day: "2-digit" }),
    profile, label: PROFILE_LABELS[profile],
    score: report.score, symmetry: report.symmetryPct, rhythm: report.rhythmPct,
    stability: report.stabilityPct, weight: report.weightTransferPct,
    speed: features.gait_speed, cadence: features.cadence,
    risk: report.fallRisk.level, riskScore: Math.round(report.fallRisk.score * 100),
    patterns: report.abnormalPatterns, recommendation: report.fallRisk.recommendation,
  };
}

function encode(s: Snapshot): string {
  try { return btoa(unescape(encodeURIComponent(JSON.stringify(s)))); } catch { return ""; }
}
function decode(b64: string): Snapshot | null {
  try { return JSON.parse(decodeURIComponent(escape(atob(b64)))) as Snapshot; } catch { return null; }
}

// ── Gemma streaming hook ────────────────────────────────────────
function useGemmaReport() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const generate = useCallback(async (snap: Snapshot) => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    setText(""); setError(null); setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/report/generate`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile: snap.profile, profile_label: snap.label, score: snap.score,
          symmetry: snap.symmetry, rhythm: snap.rhythm, stability: snap.stability,
          weight: snap.weight, speed: snap.speed, cadence: snap.cadence,
          risk: snap.risk, risk_score: snap.riskScore, patterns: snap.patterns, date: snap.date,
        }),
        signal: abortRef.current.signal,
      });
      if (!res.ok || !res.body) { setError("API 응답 오류"); setLoading(false); return; }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n"); buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6);
          if (raw === "[DONE]") { setLoading(false); return; }
          try {
            const p = JSON.parse(raw) as { token?: string; error?: string };
            if (p.error) { setError(p.error); setLoading(false); return; }
            if (p.token) setText((prev) => prev + p.token);
          } catch { /* skip */ }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError")
        setError("Ollama 연결 실패 — 로컬 서버가 실행 중인지 확인하세요 (ollama serve)");
    } finally { setLoading(false); }
  }, []);

  return { text, loading, error, generate };
}

// ── Score arc ───────────────────────────────────────────────────
function ScoreArc({ score, size = 148 }: { score: number; size?: number }) {
  const cx = size / 2, cy = size / 2;
  const R = size * 0.4, SW = size * 0.068;
  const circ = 2 * Math.PI * R;
  const dash = (score / 100) * circ;
  const c = score >= 80 ? "#10b981" : score >= 60 ? "#3b82f6" : score >= 40 ? "#f59e0b" : "#ef4444";
  const grade = score >= 80 ? "A" : score >= 60 ? "B" : score >= 40 ? "C" : "D";
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={cx} cy={cy} r={R} fill="none" stroke="rgba(148,163,184,0.10)" strokeWidth={SW} />
      <circle cx={cx} cy={cy} r={R} fill="none" stroke={c} strokeWidth={SW}
        strokeDasharray={`${dash} ${circ - dash}`} strokeDashoffset={circ / 4} strokeLinecap="round"
        style={{ transition: "stroke-dasharray 0.8s cubic-bezier(.4,0,.2,1)" }}
      />
      <text x={cx} y={cy - 6} textAnchor="middle" fill={c} fontSize={size * 0.22} fontWeight="bold" fontFamily="monospace">{score}</text>
      <text x={cx} y={cy + 10} textAnchor="middle" fill="rgba(148,163,184,0.7)" fontSize={size * 0.072} fontFamily="monospace">/ 100</text>
      <text x={cx} y={cy + 24} textAnchor="middle" fill={c} fontSize={size * 0.076} fontWeight="bold" fontFamily="monospace" opacity={0.7}>Grade {grade}</text>
    </svg>
  );
}

// ── Metric card ─────────────────────────────────────────────────
function MetricCard({ label, value, color, icon }: { label: string; value: number; color: string; icon: string }) {
  return (
    <div className="bg-bg/60 rounded-xl p-3.5 border border-border/60 relative overflow-hidden">
      <div className="absolute left-0 top-0 bottom-0 w-0.5 rounded-l-xl" style={{ background: color }} />
      <div className="flex items-start justify-between mb-2.5">
        <span className="text-textMuted text-[11px]">{label}</span>
        <span className="text-[14px]">{icon}</span>
      </div>
      <div className="text-textPri font-bold text-[20px] tabular-nums mb-2">
        {value}<span className="text-textMuted text-[12px] font-normal ml-0.5">%</span>
      </div>
      <div className="h-1.5 rounded-full bg-border/40 overflow-hidden">
        <div className="h-full rounded-full transition-all duration-1000"
          style={{ width: `${value}%`, background: color, boxShadow: `0 0 6px ${color}50` }} />
      </div>
    </div>
  );
}

// ── QR scan-frame decoration ────────────────────────────────────
function ScanFrame({ children }: { children: React.ReactNode }) {
  const c = "rgba(59,130,246,0.45)";
  const W = 14, T = 2.5;
  const corners = [
    { top: 0, left: 0, bt: T, bl: T },
    { top: 0, right: 0, bt: T, br: T },
    { bottom: 0, left: 0, bb: T, bl: T },
    { bottom: 0, right: 0, bb: T, br: T },
  ];
  return (
    <div className="relative inline-block p-3">
      {corners.map(({ top, left, right, bottom, bt, br, bb, bl }, i) => (
        <div key={i} className="absolute pointer-events-none" style={{
          top, left, right, bottom, width: W, height: W,
          borderTop:    bt ? `${bt}px solid ${c}` : undefined,
          borderBottom: bb ? `${bb}px solid ${c}` : undefined,
          borderLeft:   bl ? `${bl}px solid ${c}` : undefined,
          borderRight:  br ? `${br}px solid ${c}` : undefined,
        }} />
      ))}
      {children}
    </div>
  );
}

// ── Empty state ─────────────────────────────────────────────────
function EmptyState({ onGenerate, profile, setProfile }: {
  onGenerate: () => void;
  profile: MockProfile;
  setProfile: (p: MockProfile) => void;
}) {
  return (
    <div className="h-full flex items-center justify-center px-6">
      <div className="flex flex-col items-center gap-7 max-w-xs w-full text-center">
        <div className="relative">
          <div className="w-32 h-32 rounded-3xl bg-card border border-border flex items-center justify-center">
            <svg width={64} height={64} viewBox="0 0 72 72" fill="none" className="text-textMuted/35">
              <rect x={4} y={4} width={28} height={28} rx={3} stroke="currentColor" strokeWidth={2} />
              <rect x={40} y={4} width={28} height={28} rx={3} stroke="currentColor" strokeWidth={2} />
              <rect x={4} y={40} width={28} height={28} rx={3} stroke="currentColor" strokeWidth={2} />
              <rect x={10} y={10} width={16} height={16} rx={1} fill="currentColor" />
              <rect x={46} y={10} width={16} height={16} rx={1} fill="currentColor" />
              <rect x={10} y={46} width={16} height={16} rx={1} fill="currentColor" />
              <rect x={40} y={40} width={7} height={7} rx={1} fill="currentColor" opacity={0.4} />
              <rect x={52} y={40} width={7} height={7} rx={1} fill="currentColor" opacity={0.4} />
              <rect x={40} y={52} width={7} height={7} rx={1} fill="currentColor" opacity={0.4} />
              <rect x={52} y={52} width={7} height={7} rx={1} fill="currentColor" opacity={0.4} />
            </svg>
          </div>
          <div className="absolute inset-0 rounded-3xl border border-blue/20 animate-ping" style={{ animationDuration: "2.5s" }} />
        </div>

        <div>
          <h2 className="text-textPri font-bold text-[17px] mb-2">보행 분석 보고서 생성</h2>
          <p className="text-textMuted text-[13px] leading-relaxed">
            센서 데이터 기반 보행 분석 결과를<br />QR코드로 즉시 공유할 수 있습니다
          </p>
        </div>

        <div className="flex flex-col gap-2.5 w-full">
          <select value={profile} onChange={(e) => setProfile(e.target.value as MockProfile)}
            className="bg-card border border-border text-textSec text-[13px] rounded-xl px-4 py-3 w-full">
            {PROFILES.map((p) => <option key={p} value={p}>{PROFILE_LABELS[p]}</option>)}
          </select>
          <button onClick={onGenerate}
            className="bg-blue hover:bg-blue/85 text-white font-semibold text-[14px] px-6 py-3 rounded-xl transition-colors shadow-lg shadow-blue/20">
            보고서 생성 →
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main ────────────────────────────────────────────────────────
export default function ReportPage() {
  const [profile, setProfile] = useState<MockProfile>("normal");
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [qrUrl, setQrUrl] = useState("");
  const [copied, setCopied] = useState(false);
  const { text: aiText, loading: aiLoading, error: aiError, generate: generateAi } = useGemmaReport();

  useEffect(() => {
    const d = new URLSearchParams(window.location.search).get("d");
    if (d) { const s = decode(d); if (s) { setSnap(s); setProfile(s.profile); setQrUrl(window.location.href); } }
  }, []);

  const generate = useCallback(() => {
    const mock = generateMockSensorData(profile);
    const report = buildGaitReport(mock.features, mock.frames);
    const s = buildSnapshot(profile, report, mock.features);
    setSnap(s);
    const enc = encode(s);
    const url = `${window.location.origin}/report?d=${enc}`;
    setQrUrl(url);
    window.history.replaceState(null, "", `/report?d=${enc}`);
  }, [profile]);

  const copy = () => {
    if (!qrUrl) return;
    navigator.clipboard.writeText(qrUrl).then(() => { setCopied(true); setTimeout(() => setCopied(false), 2000); });
  };

  const rc = snap ? (RISK_COLORS[snap.risk] ?? "#6b7280") : "#6b7280";

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden">

        {/* Header — desktop only (mobile uses top bar from Sidebar) */}
        <header className="hidden md:flex bg-surface border-b border-border px-8 py-4 items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue/10 border border-blue/20 flex items-center justify-center">
              <svg width={14} height={14} viewBox="0 0 20 20" fill="none" stroke="rgb(59,130,246)" strokeWidth={1.8}>
                <rect x={2} y={2} width={7} height={7} rx={1} />
                <rect x={11} y={2} width={7} height={7} rx={1} />
                <rect x={2} y={11} width={7} height={7} rx={1} />
                <rect x={11} y={11} width={7} height={7} rx={1} />
              </svg>
            </div>
            <div>
              <h1 className="text-textPri font-semibold text-[15px]">보행패턴 결과 보고서</h1>
              <p className="text-textMuted text-[11px]">QR 공유 · Gemma AI 의견서</p>
            </div>
          </div>
          {snap && (
            <div className="flex items-center gap-2">
              <select value={profile} onChange={(e) => setProfile(e.target.value as MockProfile)}
                className="bg-card border border-border text-textSec text-[12px] rounded-lg px-3 py-1.5">
                {PROFILES.map((p) => <option key={p} value={p}>{PROFILE_LABELS[p]}</option>)}
              </select>
              <button onClick={generate}
                className="bg-blue/10 hover:bg-blue/20 text-blue border border-blue/20 font-semibold text-[12px] px-4 py-1.5 rounded-lg transition-colors">
                재생성
              </button>
            </div>
          )}
        </header>

        {/* Mobile: offset for top bar */}
        <div className="md:hidden h-12 shrink-0" />

        {/* Body */}
        <div className="flex-1 overflow-y-auto
          pb-20 md:pb-6
          px-4 py-4 md:px-6 md:py-6">

          {!snap ? (
            <EmptyState onGenerate={generate} profile={profile} setProfile={setProfile} />
          ) : (
            <div className="max-w-5xl mx-auto">

              {/* Mobile re-generate bar */}
              <div className="md:hidden flex items-center gap-2 mb-4">
                <select value={profile} onChange={(e) => setProfile(e.target.value as MockProfile)}
                  className="flex-1 bg-card border border-border text-textSec text-[12px] rounded-xl px-3 py-2.5">
                  {PROFILES.map((p) => <option key={p} value={p}>{PROFILE_LABELS[p]}</option>)}
                </select>
                <button onClick={generate}
                  className="bg-blue/10 text-blue border border-blue/20 font-semibold text-[12px] px-4 py-2.5 rounded-xl whitespace-nowrap">
                  재생성
                </button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-5">

                {/* ── Report (full → 2/3) ── */}
                <div className="lg:col-span-2 space-y-4">

                  {/* Hero title card */}
                  <div className="rounded-2xl border border-border overflow-hidden">
                    <div className="h-1 w-full" style={{ background: `linear-gradient(90deg, #3b82f6, #8b5cf6, ${rc})` }} />
                    <div className="bg-card p-4 sm:p-6">
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <div className="text-[10px] font-mono tracking-[0.12em] text-textMuted uppercase mb-2">Shoealls · 보행 분석 보고서</div>
                          <h2 className="text-textPri text-xl font-bold mb-1 truncate">{snap.label}</h2>
                          <p className="text-textSec text-[12px]">{snap.date} · 스마트슈즈 센서</p>
                          <div className="flex flex-wrap gap-1.5 mt-3">
                            {[
                              { label: "보행속도", value: `${snap.speed.toFixed(2)} m/s`, color: "#3b82f6" },
                              { label: "케이던스", value: `${snap.cadence} spm`, color: "#8b5cf6" },
                              { label: "낙상위험", value: `${snap.riskScore}%`, color: rc },
                            ].map((b) => (
                              <span key={b.label} className="text-[10px] sm:text-[11px] px-2 py-0.5 rounded-lg font-mono border"
                                style={{ color: b.color, background: `${b.color}12`, borderColor: `${b.color}30` }}>
                                {b.label} <strong>{b.value}</strong>
                              </span>
                            ))}
                          </div>
                        </div>
                        {/* Score arc — smaller on mobile */}
                        <div className="shrink-0">
                          <ScoreArc score={snap.score} size={110} />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* 2×2 Metric grid */}
                  <div>
                    <p className="text-textMuted text-[11px] uppercase tracking-wider mb-2 px-0.5">세부 지표</p>
                    <div className="grid grid-cols-2 gap-2.5 sm:gap-3">
                      <MetricCard label="좌우 대칭성" value={snap.symmetry} color="#3b82f6" icon="⇌" />
                      <MetricCard label="보행 리듬"   value={snap.rhythm}   color="#8b5cf6" icon="♩" />
                      <MetricCard label="동적 안정성" value={snap.stability} color="#10b981" icon="◎" />
                      <MetricCard label="체중 이동"   value={snap.weight}   color="#f59e0b" icon="⟳" />
                    </div>
                  </div>

                  {/* Patterns + Risk */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="bg-card rounded-xl border border-border p-4">
                      <div className="flex items-center gap-1.5 mb-3">
                        <div className="w-1 h-4 rounded-full bg-amber" />
                        <span className="text-textPri font-semibold text-[12px]">이상 패턴</span>
                      </div>
                      {snap.patterns.length === 0 ? (
                        <div className="flex items-center gap-2 text-green text-[12px]">
                          <svg width={13} height={13} viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth={2}><polyline points="2,8 6,12 13,3" /></svg>
                          이상 패턴 없음
                        </div>
                      ) : (
                        <ul className="space-y-1.5">
                          {snap.patterns.map((p) => (
                            <li key={p} className="flex items-start gap-2 text-textSec text-[12px]">
                              <span className="w-1 h-1 rounded-full bg-amber shrink-0 mt-1.5" />
                              {p}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>

                    <div className="rounded-xl border p-4" style={{ background: RISK_BG[snap.risk], borderColor: `${rc}30` }}>
                      <div className="flex items-center gap-1.5 mb-3">
                        <div className="w-1 h-4 rounded-full" style={{ background: rc }} />
                        <span className="font-semibold text-[12px]" style={{ color: rc }}>낙상 위험도</span>
                        <span className="ml-auto text-[18px] font-bold tabular-nums" style={{ color: rc }}>{snap.riskScore}%</span>
                      </div>
                      <div className="h-1 rounded-full bg-border/40 mb-3 overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${snap.riskScore}%`, background: rc }} />
                      </div>
                      <p className="text-textSec text-[11px] leading-relaxed">{snap.recommendation}</p>
                    </div>
                  </div>

                  {/* ── QR card (mobile only — inline) ── */}
                  <div className="lg:hidden bg-card rounded-2xl border border-border overflow-hidden">
                    <div className="px-4 py-3 border-b border-border/60 flex items-center gap-2">
                      <svg width={13} height={13} viewBox="0 0 15 15" fill="none" stroke="rgb(59,130,246)" strokeWidth={1.6}>
                        <rect x={1} y={1} width={5} height={5} rx={1} /><rect x={9} y={1} width={5} height={5} rx={1} />
                        <rect x={1} y={9} width={5} height={5} rx={1} />
                        <rect x={9} y={9} width={2.5} height={2.5} /><rect x={12.5} y={9} width={2.5} height={2.5} />
                        <rect x={9} y={12.5} width={2.5} height={2.5} /><rect x={12.5} y={12.5} width={2.5} height={2.5} />
                      </svg>
                      <span className="text-textPri font-semibold text-[13px]">QR 공유</span>
                    </div>
                    <div className="p-4 flex flex-col sm:flex-row items-center gap-5">
                      <ScanFrame>
                        <div className="bg-white rounded-xl p-3">
                          <QRCodeSVG value={qrUrl} size={140} bgColor="#ffffff" fgColor="#0f172a" level="M" marginSize={1} />
                        </div>
                      </ScanFrame>
                      <div className="flex-1 w-full space-y-3">
                        <div>
                          <p className="text-textSec text-[12px] font-medium mb-1.5">스캔하여 보고서 공유</p>
                          <p className="text-textMuted text-[10px] font-mono bg-bg rounded-lg px-2 py-1.5 break-all leading-relaxed">
                            {qrUrl.replace("https://", "").replace("http://", "")}
                          </p>
                        </div>
                        <div className="flex gap-2">
                          <button onClick={copy}
                            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2.5 rounded-xl border text-[12px] font-medium transition-all"
                            style={copied ? { background: "rgba(16,185,129,0.08)", borderColor: "rgba(16,185,129,0.25)", color: "#10b981" } : {}}>
                            {copied ? "✓ 복사됨" : "URL 복사"}
                          </button>
                          <button onClick={() => window.print()}
                            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2.5 rounded-xl text-[12px] font-medium"
                            style={{ background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.2)", color: "#3b82f6" }}>
                            인쇄 / PDF
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Gemma AI section */}
                  <div className="bg-card rounded-2xl border border-border overflow-hidden">
                    <div className="flex items-center justify-between px-4 sm:px-5 py-3.5 border-b border-border/60"
                      style={{ background: "linear-gradient(135deg, rgba(139,92,246,0.05) 0%, transparent 60%)" }}>
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-md bg-purple/15 border border-purple/20 flex items-center justify-center">
                          <span className="text-purple text-[11px] font-bold">G</span>
                        </div>
                        <div className="min-w-0">
                          <span className="text-textPri font-semibold text-[13px]">Gemma AI 의견서</span>
                          <span className="hidden sm:inline ml-2 text-purple text-[10px] font-mono">gemma2:2b</span>
                        </div>
                      </div>
                      <button onClick={() => generateAi(snap)} disabled={aiLoading}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold text-[12px] transition-all disabled:opacity-50 whitespace-nowrap shrink-0"
                        style={{ background: "rgba(139,92,246,0.12)", border: "1px solid rgba(139,92,246,0.25)", color: "#af65fa" }}>
                        {aiLoading ? (
                          <><svg className="animate-spin" width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
                            <circle cx={12} cy={12} r={10} strokeOpacity={0.2} /><path d="M12 2a10 10 0 0 1 10 10" />
                          </svg> 생성 중…</>
                        ) : "AI 리포트 생성"}
                      </button>
                    </div>
                    <div className="px-4 sm:px-5 py-4">
                      {aiError && (
                        <div className="text-[12px] text-red bg-red/8 rounded-lg px-3 py-2.5 border border-red/15 mb-3">{aiError}</div>
                      )}
                      {!aiText && !aiLoading && !aiError && (
                        <p className="text-textMuted text-[13px] leading-relaxed py-1">
                          버튼을 클릭하면 로컬 Gemma 모델이 보행 분석 결과를 해석하여 전문 의견서를 실시간으로 작성합니다.
                          <span className="text-textMuted/55 text-[11px] mt-1 block">
                            사전 조건: <code className="font-mono text-purple/60">ollama serve</code> · <code className="font-mono text-purple/60">gemma2:2b</code>
                          </span>
                        </p>
                      )}
                      {(aiText || aiLoading) && (
                        <div className="text-textSec text-[13px] leading-7 whitespace-pre-wrap min-h-[60px]">
                          {aiText}
                          {aiLoading && <span className="inline-block w-0.5 h-[1em] bg-purple ml-0.5 align-text-bottom animate-pulse" />}
                        </div>
                      )}
                      <p className="text-textMuted/45 text-[10px] mt-3 pt-3 border-t border-border/40">
                        ※ AI 의견서는 참고용이며 의학적 진단을 대체하지 않습니다.
                      </p>
                    </div>
                  </div>

                </div>

                {/* ── QR panel (desktop sidebar — hidden on mobile) ── */}
                <div className="hidden lg:block space-y-4">
                  <div className="bg-card rounded-2xl border border-border overflow-hidden sticky top-4">
                    <div className="px-5 py-3.5 border-b border-border/60 flex items-center gap-2">
                      <svg width={13} height={13} viewBox="0 0 15 15" fill="none" stroke="rgb(59,130,246)" strokeWidth={1.6}>
                        <rect x={1} y={1} width={5} height={5} rx={1} /><rect x={9} y={1} width={5} height={5} rx={1} />
                        <rect x={1} y={9} width={5} height={5} rx={1} />
                        <rect x={9} y={9} width={2.5} height={2.5} /><rect x={12.5} y={9} width={2.5} height={2.5} />
                        <rect x={9} y={12.5} width={2.5} height={2.5} /><rect x={12.5} y={12.5} width={2.5} height={2.5} />
                      </svg>
                      <span className="text-textPri font-semibold text-[13px]">QR 공유</span>
                    </div>
                    <div className="p-5 flex flex-col items-center gap-4">
                      <ScanFrame>
                        <div className="bg-white rounded-xl p-3 shadow-inner">
                          <QRCodeSVG value={qrUrl} size={166} bgColor="#ffffff" fgColor="#0f172a" level="M" marginSize={1} />
                        </div>
                      </ScanFrame>
                      <div className="text-center w-full">
                        <p className="text-textSec text-[12px] font-medium">스캔하여 보고서 공유</p>
                        <p className="text-textMuted text-[10px] font-mono mt-1.5 bg-bg rounded-lg px-2 py-1.5 break-all leading-relaxed line-clamp-2">
                          {qrUrl.replace("https://", "").replace("http://", "")}
                        </p>
                      </div>
                      <div className="w-full h-px bg-border/50" />
                      <div className="w-full space-y-2">
                        <button onClick={copy}
                          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border text-[12px] font-medium transition-all"
                          style={copied
                            ? { background: "rgba(16,185,129,0.08)", borderColor: "rgba(16,185,129,0.25)", color: "#10b981" }
                            : { background: "transparent", borderColor: "", color: "" }}>
                          {copied ? (
                            <><svg width={13} height={13} viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth={2}><polyline points="2,8 6,12 13,3" /></svg> 복사 완료</>
                          ) : (
                            <><svg width={13} height={13} viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth={1.5}>
                              <rect x={4} y={4} width={9} height={9} rx={1.5} /><path d="M2 11V2h9" />
                            </svg> URL 복사</>
                          )}
                        </button>
                        <button onClick={() => window.print()}
                          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-[12px] font-medium transition-all"
                          style={{ background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.2)", color: "#3b82f6" }}>
                          <svg width={13} height={13} viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth={1.5}>
                            <rect x={3} y={1} width={9} height={4} rx={1} />
                            <path d="M3 5H1a1 1 0 00-1 1v5a1 1 0 001 1h2v-2h9v2h2a1 1 0 001-1V6a1 1 0 00-1-1h-2" />
                            <rect x={3} y={10} width={9} height={4} rx={1} />
                          </svg>
                          인쇄 / PDF 저장
                        </button>
                      </div>
                      <p className="text-textMuted/45 text-[10px] text-center leading-relaxed">
                        보행 지표만 포함 · 외부 전송 없음
                      </p>
                    </div>
                  </div>
                </div>

              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
