"use client";

import { useCallback, useEffect, useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import Sidebar from "@/components/Sidebar";
import { buildGaitReport } from "@/lib/gaitAnalysis";
import { generateMockSensorData, PROFILE_LABELS, type MockProfile } from "@/lib/mockSensorData";
import type { GaitFeatures, GaitReport } from "@/types/sensor";

const PROFILES: MockProfile[] = ["normal", "parkinsons", "stroke", "fall_risk"];

const RISK_COLORS: Record<string, string> = {
  low:      "#10b981",
  watch:    "#3b82f6",
  elevated: "#f59e0b",
  high:     "#ef4444",
};
const RISK_KR: Record<string, string> = {
  low: "낮음", watch: "관찰", elevated: "주의", high: "높음",
};

interface Snapshot {
  date: string;
  profile: MockProfile;
  label: string;
  score: number;
  symmetry: number;
  rhythm: number;
  stability: number;
  weight: number;
  speed: number;
  cadence: number;
  risk: string;
  riskScore: number;
  patterns: string[];
  recommendation: string;
}

function buildSnapshot(profile: MockProfile, report: GaitReport, features: GaitFeatures): Snapshot {
  return {
    date: new Date().toLocaleDateString("ko-KR", { year: "numeric", month: "2-digit", day: "2-digit" }),
    profile,
    label: PROFILE_LABELS[profile],
    score: report.score,
    symmetry: report.symmetryPct,
    rhythm: report.rhythmPct,
    stability: report.stabilityPct,
    weight: report.weightTransferPct,
    speed: features.gait_speed,
    cadence: features.cadence,
    risk: report.fallRisk.level,
    riskScore: Math.round(report.fallRisk.score * 100),
    patterns: report.abnormalPatterns,
    recommendation: report.fallRisk.recommendation,
  };
}

function encode(s: Snapshot): string {
  try { return btoa(unescape(encodeURIComponent(JSON.stringify(s)))); }
  catch { return ""; }
}

function decode(b64: string): Snapshot | null {
  try { return JSON.parse(decodeURIComponent(escape(atob(b64)))) as Snapshot; }
  catch { return null; }
}

// ── Score ring ───────────────────────────────────────────────────
function ScoreRing({ score }: { score: number }) {
  const R = 52, SW = 9;
  const circ = 2 * Math.PI * R;
  const dash = (score / 100) * circ;
  const color = score >= 80 ? "#10b981" : score >= 60 ? "#3b82f6" : score >= 40 ? "#f59e0b" : "#ef4444";
  return (
    <svg width={126} height={126} viewBox="0 0 126 126">
      <circle cx={63} cy={63} r={R} fill="none" stroke="rgba(148,163,184,0.12)" strokeWidth={SW} />
      <circle cx={63} cy={63} r={R} fill="none" stroke={color} strokeWidth={SW}
        strokeDasharray={`${dash} ${circ - dash}`}
        strokeDashoffset={circ / 4} strokeLinecap="round"
        style={{ transition: "stroke-dasharray 0.6s ease" }}
      />
      <text x={63} y={58} textAnchor="middle" fill="rgb(241 245 249)" fontSize={26} fontWeight="bold" fontFamily="monospace">{score}</text>
      <text x={63} y={75} textAnchor="middle" fill="rgb(148 163 184)" fontSize={10} fontFamily="monospace">종합점수</text>
    </svg>
  );
}

// ── Metric bar ──────────────────────────────────────────────────
function Bar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-[12px] mb-1">
        <span className="text-textSec">{label}</span>
        <span className="text-textPri font-semibold tabular-nums">{value}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-border/40 overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${value}%`, background: color }} />
      </div>
    </div>
  );
}

// ── QR placeholder icon ─────────────────────────────────────────
function QrPlaceholder() {
  return (
    <svg width={56} height={56} viewBox="0 0 56 56" fill="none" stroke="currentColor" strokeWidth={1.5} className="text-textMuted">
      <rect x={4} y={4} width={20} height={20} rx={2} />
      <rect x={32} y={4} width={20} height={20} rx={2} />
      <rect x={4} y={32} width={20} height={20} rx={2} />
      <rect x={9} y={9} width={10} height={10} fill="currentColor" stroke="none" />
      <rect x={37} y={9} width={10} height={10} fill="currentColor" stroke="none" />
      <rect x={9} y={37} width={10} height={10} fill="currentColor" stroke="none" />
      <path d="M32 32h6v6h-6z M44 32h6v6h-6z M32 44h6v6h-6z M44 44h6v6h-6z" fill="currentColor" stroke="none" />
    </svg>
  );
}

// ── Main ────────────────────────────────────────────────────────
export default function ReportPage() {
  const [profile, setProfile] = useState<MockProfile>("normal");
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [qrUrl, setQrUrl] = useState("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const d = new URLSearchParams(window.location.search).get("d");
    if (d) {
      const s = decode(d);
      if (s) { setSnap(s); setProfile(s.profile); setQrUrl(window.location.href); }
    }
  }, []);

  const generate = useCallback(() => {
    const mock = generateMockSensorData(profile);
    const report = buildGaitReport(mock.features, mock.frames);
    const s = buildSnapshot(profile, report, mock.features);
    setSnap(s);
    const encoded = encode(s);
    const url = `${window.location.origin}/report?d=${encoded}`;
    setQrUrl(url);
    window.history.replaceState(null, "", `/report?d=${encoded}`);
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

        {/* Header */}
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-textPri font-semibold text-xl">보행패턴 결과 보고서</h1>
            <p className="text-textMuted text-[12px] mt-0.5">생성 후 QR코드로 즉시 공유</p>
          </div>
          <div className="flex items-center gap-3">
            <select value={profile} onChange={(e) => setProfile(e.target.value as MockProfile)}
              className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5">
              {PROFILES.map((p) => <option key={p} value={p}>{PROFILE_LABELS[p]}</option>)}
            </select>
            <button onClick={generate}
              className="bg-blue hover:bg-blue/80 text-white font-semibold text-[13px] px-5 py-2 rounded-lg transition-colors">
              보고서 생성
            </button>
          </div>
        </header>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-8">
          {!snap ? (
            <div className="h-full flex flex-col items-center justify-center gap-3 text-textMuted">
              <QrPlaceholder />
              <p className="text-[14px]">프로파일을 선택하고 보고서를 생성하세요</p>
            </div>
          ) : (
            <div className="max-w-5xl mx-auto grid grid-cols-3 gap-6">

              {/* ── Report (2/3) ── */}
              <div className="col-span-2 space-y-5">

                {/* Title card */}
                <div className="bg-card rounded-xl border border-border p-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="text-textMuted text-[10px] uppercase tracking-widest mb-1.5">Shoealls 보행 분석 보고서</div>
                      <div className="text-textPri text-xl font-bold">{snap.label}</div>
                      <div className="text-textSec text-[13px] mt-1">{snap.date} · 스마트슈즈 센서 기반</div>
                      <div className="flex gap-2 mt-3">
                        <span className="text-[11px] px-2 py-0.5 rounded-full bg-blue/10 text-blue border border-blue/20">
                          보행속도 {snap.speed.toFixed(2)} m/s
                        </span>
                        <span className="text-[11px] px-2 py-0.5 rounded-full bg-purple/10 text-purple border border-purple/20">
                          케이던스 {snap.cadence} spm
                        </span>
                      </div>
                    </div>
                    <ScoreRing score={snap.score} />
                  </div>
                </div>

                {/* Metric bars */}
                <div className="bg-card rounded-xl border border-border p-5 space-y-4">
                  <div className="text-textPri font-semibold text-[13px]">세부 지표</div>
                  <Bar label="좌우 대칭성" value={snap.symmetry} color="#3b82f6" />
                  <Bar label="보행 리듬" value={snap.rhythm}   color="#8b5cf6" />
                  <Bar label="동적 안정성" value={snap.stability} color="#10b981" />
                  <Bar label="체중 이동" value={snap.weight}   color="#f59e0b" />
                </div>

                {/* Key values grid */}
                <div className="bg-card rounded-xl border border-border p-5">
                  <div className="text-textPri font-semibold text-[13px] mb-3">주요 수치</div>
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { label: "보행 속도", value: `${snap.speed.toFixed(2)} m/s`,       sub: snap.speed >= 1.0 ? "정상 범위" : "저하 감지" },
                      { label: "케이던스",  value: `${snap.cadence} steps/min`,           sub: snap.cadence >= 100 && snap.cadence <= 130 ? "정상 범위" : "범위 이탈" },
                      { label: "낙상 위험", value: `${snap.riskScore}%`,                  sub: RISK_KR[snap.risk] },
                      { label: "이상 패턴", value: `${snap.patterns.length}건 감지`,      sub: snap.patterns.length === 0 ? "이상 없음" : "주의 필요" },
                    ].map((item) => (
                      <div key={item.label} className="bg-bg/50 rounded-lg p-3">
                        <div className="text-textMuted text-[11px]">{item.label}</div>
                        <div className="text-textPri font-bold text-[15px] mt-0.5 tabular-nums">{item.value}</div>
                        <div className="text-[10px] mt-0.5" style={{ color: rc }}>{item.sub}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Abnormal patterns */}
                {snap.patterns.length > 0 && (
                  <div className="bg-amber/8 rounded-xl border border-amber/25 p-5">
                    <div className="text-amber font-semibold text-[13px] mb-2">이상 패턴 감지</div>
                    <ul className="space-y-1.5">
                      {snap.patterns.map((p) => (
                        <li key={p} className="flex items-center gap-2 text-textSec text-[13px]">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber shrink-0" />
                          {p}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Risk + recommendation */}
                <div className="rounded-xl border p-5" style={{ background: `${rc}0d`, borderColor: `${rc}30` }}>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: rc }} />
                    <span className="font-semibold text-[13px]" style={{ color: rc }}>
                      낙상 위험도 · {RISK_KR[snap.risk]}
                    </span>
                  </div>
                  <p className="text-textSec text-[13px] leading-relaxed">{snap.recommendation}</p>
                  <p className="text-textMuted text-[11px] mt-3">
                    ※ 본 보고서는 스마트슈즈 센서 데이터 기반 참고 지표이며 의학적 진단이 아닙니다.
                  </p>
                </div>
              </div>

              {/* ── QR (1/3) ── */}
              <div className="space-y-4">
                <div className="bg-card rounded-xl border border-border p-5 flex flex-col items-center gap-4 sticky top-0">
                  <div className="text-textPri font-semibold text-[13px] self-start">QR 코드 공유</div>

                  {/* QR code */}
                  <div className="bg-white rounded-2xl p-3.5 shadow-lg ring-1 ring-border/20">
                    <QRCodeSVG
                      value={qrUrl}
                      size={176}
                      bgColor="#ffffff"
                      fgColor="#0f172a"
                      level="M"
                      marginSize={1}
                    />
                  </div>

                  <div className="text-center space-y-1 w-full">
                    <div className="text-textSec text-[12px]">스캔하여 보고서 공유</div>
                    <div className="text-textMuted text-[10px] font-mono bg-bg rounded-lg px-2 py-1.5 break-all line-clamp-2 leading-relaxed">
                      {qrUrl.replace("https://", "").replace("http://", "")}
                    </div>
                  </div>

                  <button onClick={copy}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg border border-border text-textSec hover:bg-bg hover:text-textPri transition-colors text-[12px]">
                    {copied ? (
                      <><svg width={12} height={12} viewBox="0 0 15 15" fill="none" stroke="#10b981" strokeWidth={2}><polyline points="2,8 6,12 13,3" /></svg>
                        <span className="text-green">복사 완료</span></>
                    ) : (
                      <><svg width={12} height={12} viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth={1.4}>
                        <rect x={4} y={4} width={9} height={9} rx={1} /><path d="M2 11V2h9" />
                      </svg>URL 복사</>
                    )}
                  </button>

                  <button onClick={() => window.print()}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-blue/10 border border-blue/20 text-blue hover:bg-blue/20 transition-colors text-[12px]">
                    <svg width={12} height={12} viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth={1.4}>
                      <rect x={2} y={1} width={11} height={5} rx={1} />
                      <path d="M2 6H0a1 1 0 00-1 1v5a1 1 0 001 1h2v-3h11v3h2a1 1 0 001-1V7a1 1 0 00-1-1h-2" />
                      <rect x={2} y={10} width={11} height={4} rx={1} />
                    </svg>
                    인쇄 / PDF 저장
                  </button>

                  <p className="text-textMuted text-[10px] text-center leading-relaxed border-t border-border/60 pt-3">
                    QR코드는 보행 지표만 포함하며 외부 서버로 데이터가 전송되지 않습니다.
                  </p>
                </div>
              </div>

            </div>
          )}
        </div>
      </main>
    </div>
  );
}
