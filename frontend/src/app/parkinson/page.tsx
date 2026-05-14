"use client";

import { useCallback, useState } from "react";
import Sidebar from "@/components/Sidebar";
import { MockProfile, PROFILE_LABELS } from "@/lib/mockSensorData";

/* ─── Types & Constants ──────────────────────────────────────────────────────── */

const PROFILES: MockProfile[] = ["normal", "parkinsons", "stroke", "fall_risk"];

interface Biomarker {
  id: string;
  name: string;
  nameEn: string;
  unit: string;
  normalRange: string;
  values: Record<MockProfile, { display: string; score: number }>;
}

// score: 0 = normal, 1 = max abnormal
const BIOMARKERS: Biomarker[] = [
  {
    id: "speed", name: "보행 속도", nameEn: "Gait Speed", unit: "m/s", normalRange: "> 1.0",
    values: {
      normal:     { display: "1.22", score: 0.04 },
      parkinsons: { display: "0.68", score: 0.82 },
      stroke:     { display: "0.52", score: 0.91 },
      fall_risk:  { display: "0.72", score: 0.42 },
    },
  },
  {
    id: "cadence", name: "케이던스 패턴", nameEn: "Festination Index", unit: "spm", normalRange: "< 120",
    values: {
      normal:     { display: "112",  score: 0.05 },
      parkinsons: { display: "144",  score: 0.78 },
      stroke:     { display: "98",   score: 0.18 },
      fall_risk:  { display: "118",  score: 0.28 },
    },
  },
  {
    id: "symmetry", name: "보폭 대칭성", nameEn: "Step Symmetry", unit: "%", normalRange: "> 90%",
    values: {
      normal:     { display: "93",   score: 0.05 },
      parkinsons: { display: "79",   score: 0.62 },
      stroke:     { display: "58",   score: 0.92 },
      fall_risk:  { display: "82",   score: 0.46 },
    },
  },
  {
    id: "regularity", name: "보폭 규칙성", nameEn: "Stride Regularity", unit: "%", normalRange: "> 80%",
    values: {
      normal:     { display: "88",   score: 0.06 },
      parkinsons: { display: "54",   score: 0.79 },
      stroke:     { display: "68",   score: 0.38 },
      fall_risk:  { display: "56",   score: 0.56 },
    },
  },
  {
    id: "tremor", name: "떨림 지수", nameEn: "Tremor Index (4–6 Hz)", unit: "a.u.", normalRange: "< 0.2",
    values: {
      normal:     { display: "0.04", score: 0.04 },
      parkinsons: { display: "0.72", score: 0.88 },
      stroke:     { display: "0.08", score: 0.09 },
      fall_risk:  { display: "0.18", score: 0.22 },
    },
  },
  {
    id: "sway", name: "자세 흔들림", nameEn: "Postural Sway (CoP)", unit: "cm", normalRange: "< 3.0",
    values: {
      normal:     { display: "2.1",  score: 0.08 },
      parkinsons: { display: "5.8",  score: 0.62 },
      stroke:     { display: "7.2",  score: 0.78 },
      fall_risk:  { display: "8.4",  score: 0.85 },
    },
  },
  {
    id: "heeltoe", name: "발뒤꿈치-발끝 전이", nameEn: "Heel-Toe Ratio", unit: "%", normalRange: "> 70%",
    values: {
      normal:     { display: "76",   score: 0.07 },
      parkinsons: { display: "42",   score: 0.73 },
      stroke:     { display: "55",   score: 0.44 },
      fall_risk:  { display: "61",   score: 0.34 },
    },
  },
  {
    id: "asymmetry", name: "좌우 비대칭", nameEn: "Lateral Asymmetry", unit: "index", normalRange: "< 0.15",
    values: {
      normal:     { display: "0.06", score: 0.06 },
      parkinsons: { display: "0.31", score: 0.64 },
      stroke:     { display: "0.58", score: 0.95 },
      fall_risk:  { display: "0.24", score: 0.50 },
    },
  },
];

interface DiagnosisResult {
  stage: 0 | 1 | 2 | 3;
  stageName: string;
  stageEn: string;
  confidence: number;
  summary: string;
  isParkinson: boolean;
  recommendations: string[];
}

const DIAGNOSES: Record<MockProfile, DiagnosisResult> = {
  normal: {
    stage: 0, stageName: "정상", stageEn: "Normal", confidence: 0.97, isParkinson: false,
    summary: "파킨슨 관련 보행 이상 징후가 감지되지 않았습니다. 모든 바이오마커가 정상 범위 내에 있습니다.",
    recommendations: ["정기 모니터링 유지 (3개월 주기)", "유산소 운동 지속 권고", "6개월 추적 보행 분석 권고"],
  },
  parkinsons: {
    stage: 2, stageName: "초기 의심", stageEn: "Early PD Suspect", confidence: 0.87, isParkinson: true,
    summary: "파킨슨병 초기 보행 패턴과 높은 일치도가 확인됩니다. 떨림 지수·케이던스·보폭 규칙성에서 유의미한 이상이 감지되었습니다.",
    recommendations: ["신경과 전문의 방문 권고 (즉시)", "UPDRS 임상 평가 필요", "DaTscan 또는 MRI 검사 고려", "3개월 단위 보행 추적 분석 시행", "낙상 예방 교육 및 환경 개선"],
  },
  stroke: {
    stage: 1, stageName: "비정형 이상", stageEn: "Atypical Gait", confidence: 0.91, isParkinson: false,
    summary: "편측 비대칭이 매우 두드러져 파킨슨 패턴보다는 뇌졸중 후 편마비 보행 패턴에 더 가깝습니다. 파킨슨 가능성은 낮습니다.",
    recommendations: ["재활의학과 또는 신경과 상담 권고", "뇌혈관 후유증 평가 필요", "물리치료 및 보행 재활 프로그램", "낙상 위험 매우 높음 — 즉각 개입 필요"],
  },
  fall_risk: {
    stage: 1, stageName: "전구기 징후", stageEn: "Prodromal Signs", confidence: 0.82, isParkinson: true,
    summary: "초기 파킨슨 전구 증상과 일치하는 보행 패턴이 관찰됩니다. 자세 흔들림과 보폭 불규칙성이 기준치를 초과합니다.",
    recommendations: ["신경과 상담 권고 (3개월 내)", "6개월 단위 추적 보행 분석", "균형 훈련 및 낙상 예방 운동", "가정 내 낙상 위험 환경 점검"],
  },
};

const STAGE_COLORS = ["#10B981", "#F59E0B", "#EF4444", "#7C3AED"];
const STAGE_BG     = ["#10B98115", "#F59E0B15", "#EF444415", "#7C3AED15"];

// 7-day simulated trend per profile
const TREND: Record<MockProfile, { day: string; stage: number; speed: number; tremor: number }[]> = {
  normal:     [
    { day: "5/8",  stage: 0, speed: 1.23, tremor: 0.03 },
    { day: "5/9",  stage: 0, speed: 1.21, tremor: 0.04 },
    { day: "5/10", stage: 0, speed: 1.24, tremor: 0.04 },
    { day: "5/11", stage: 0, speed: 1.22, tremor: 0.04 },
    { day: "5/12", stage: 0, speed: 1.25, tremor: 0.03 },
    { day: "5/13", stage: 0, speed: 1.23, tremor: 0.05 },
    { day: "5/14", stage: 0, speed: 1.22, tremor: 0.04 },
  ],
  parkinsons: [
    { day: "5/8",  stage: 1, speed: 0.82, tremor: 0.48 },
    { day: "5/9",  stage: 1, speed: 0.78, tremor: 0.54 },
    { day: "5/10", stage: 2, speed: 0.74, tremor: 0.61 },
    { day: "5/11", stage: 2, speed: 0.71, tremor: 0.67 },
    { day: "5/12", stage: 2, speed: 0.70, tremor: 0.70 },
    { day: "5/13", stage: 2, speed: 0.69, tremor: 0.71 },
    { day: "5/14", stage: 2, speed: 0.68, tremor: 0.72 },
  ],
  stroke:     [
    { day: "5/8",  stage: 1, speed: 0.60, tremor: 0.07 },
    { day: "5/9",  stage: 1, speed: 0.58, tremor: 0.08 },
    { day: "5/10", stage: 1, speed: 0.56, tremor: 0.08 },
    { day: "5/11", stage: 1, speed: 0.54, tremor: 0.08 },
    { day: "5/12", stage: 1, speed: 0.53, tremor: 0.09 },
    { day: "5/13", stage: 1, speed: 0.53, tremor: 0.08 },
    { day: "5/14", stage: 1, speed: 0.52, tremor: 0.08 },
  ],
  fall_risk:  [
    { day: "5/8",  stage: 0, speed: 0.88, tremor: 0.12 },
    { day: "5/9",  stage: 1, speed: 0.84, tremor: 0.15 },
    { day: "5/10", stage: 1, speed: 0.81, tremor: 0.16 },
    { day: "5/11", stage: 1, speed: 0.78, tremor: 0.17 },
    { day: "5/12", stage: 1, speed: 0.75, tremor: 0.17 },
    { day: "5/13", stage: 1, speed: 0.73, tremor: 0.18 },
    { day: "5/14", stage: 1, speed: 0.72, tremor: 0.18 },
  ],
};

/* ─── Radar Chart ────────────────────────────────────────────────────────────── */

function RadarChart({ profile, color }: { profile: MockProfile; color: string }) {
  const cx = 145, cy = 145, R = 108;
  const n = BIOMARKERS.length;

  function pt(score: number, i: number): [number, number] {
    const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
    return [cx + score * R * Math.cos(angle), cy + score * R * Math.sin(angle)];
  }

  function polyPath(scores: number[]): string {
    return scores
      .map((s, i) => { const [x, y] = pt(s, i); return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`; })
      .join(" ") + "Z";
  }

  const scores = BIOMARKERS.map((b) => b.values[profile].score);
  const normalRef = BIOMARKERS.map(() => 0.18);

  return (
    <svg width={290} height={290} style={{ overflow: "visible" }}>
      {/* Grid rings */}
      {[0.25, 0.5, 0.75, 1.0].map((r, ri) => (
        <path key={ri} d={polyPath(Array(n).fill(r))}
          fill="none" stroke="#32425B" strokeWidth={ri === 3 ? 1 : 0.6}
          strokeOpacity={ri === 3 ? 0.6 : 0.35} strokeDasharray={ri < 3 ? "3,4" : undefined} />
      ))}

      {/* Axes */}
      {BIOMARKERS.map((_, i) => {
        const [x, y] = pt(1, i);
        return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="#32425B" strokeWidth={0.7} strokeOpacity={0.5} />;
      })}

      {/* Normal reference zone */}
      <path d={polyPath(normalRef)} fill="#10B981" fillOpacity={0.08}
        stroke="#10B981" strokeWidth={1} strokeOpacity={0.25} strokeDasharray="3,4" />

      {/* Score polygon */}
      <path d={polyPath(scores)} fill={color} fillOpacity={0.18} stroke={color} strokeWidth={2} strokeOpacity={0.85} />

      {/* Score dots */}
      {scores.map((s, i) => {
        const [x, y] = pt(s, i);
        return (
          <g key={i}>
            <circle cx={x} cy={y} r={5.5} fill={color} opacity={0.25} />
            <circle cx={x} cy={y} r={3.5} fill={color} />
          </g>
        );
      })}

      {/* Axis labels */}
      {BIOMARKERS.map((b, i) => {
        const [lx, ly] = pt(1.22, i);
        const anchor = lx < cx - 4 ? "end" : lx > cx + 4 ? "start" : "middle";
        return (
          <text key={i} x={lx} y={ly + 4} textAnchor={anchor}
            fontSize={9.5} fill="#607085"
            fontFamily="ui-sans-serif, system-ui, sans-serif">
            {b.name}
          </text>
        );
      })}

      {/* Center label */}
      <text x={cx} y={cy + 4} textAnchor="middle" fontSize={9} fill="#32425B"
        fontFamily="ui-monospace, monospace">PD Risk</text>

      {/* Ring labels */}
      {[25, 50, 75].map((pct, i) => (
        <text key={i} x={cx + 4} y={cy - (i + 1) * 0.25 * R - 2}
          fontSize={8} fill="#32425B" fontFamily="ui-monospace, monospace">{pct}%</text>
      ))}
    </svg>
  );
}

/* ─── Trend Mini Chart ───────────────────────────────────────────────────────── */

function TrendChart({ profile }: { profile: MockProfile }) {
  const data = TREND[profile];
  const W = 440, H = 100;
  const pad = { l: 36, r: 12, t: 10, b: 24 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;

  // speed: 0.4–1.4, tremor: 0–1
  const speedMin = 0.4, speedMax = 1.35;
  const tremorMax = 1.0;

  function sx(i: number) { return pad.l + (i / (data.length - 1)) * innerW; }
  function speedY(v: number) { return pad.t + innerH - ((v - speedMin) / (speedMax - speedMin)) * innerH; }
  function tremorY(v: number) { return pad.t + innerH - (v / tremorMax) * innerH; }

  const speedPath = data.map((d, i) => `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)},${speedY(d.speed).toFixed(1)}`).join(" ");
  const tremorPath = data.map((d, i) => `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)},${tremorY(d.tremor).toFixed(1)}`).join(" ");

  return (
    <svg width={W} height={H}>
      {/* Grid */}
      {[0.25, 0.5, 0.75, 1.0].map((r, i) => (
        <line key={i} x1={pad.l} y1={pad.t + innerH * (1 - r)} x2={W - pad.r} y2={pad.t + innerH * (1 - r)}
          stroke="#1E2940" strokeWidth={0.8} />
      ))}

      {/* Stage highlight bands */}
      {data.map((d, i) => i < data.length - 1 && (
        <rect key={i}
          x={sx(i)} y={pad.t} width={sx(i + 1) - sx(i)} height={innerH}
          fill={STAGE_COLORS[d.stage]} opacity={0.04} />
      ))}

      {/* Speed line */}
      <path d={speedPath} fill="none" stroke="#38BDF8" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />

      {/* Tremor line */}
      <path d={tremorPath} fill="none" stroke="#F87171" strokeWidth={1.5} strokeDasharray="4,3"
        strokeLinecap="round" strokeLinejoin="round" />

      {/* Stage dots */}
      {data.map((d, i) => (
        <circle key={i} cx={sx(i)} cy={speedY(d.speed)} r={3.5}
          fill={STAGE_COLORS[d.stage]} stroke="#0A1220" strokeWidth={1.5} />
      ))}

      {/* X-axis labels */}
      {data.map((d, i) => (
        <text key={i} x={sx(i)} y={H - 6} textAnchor="middle"
          fontSize={9} fill="#607085" fontFamily="ui-monospace, monospace">{d.day}</text>
      ))}

      {/* Y-axis label */}
      <text x={pad.l - 4} y={pad.t + innerH / 2} textAnchor="middle" fontSize={8} fill="#607085"
        fontFamily="ui-monospace, monospace" transform={`rotate(-90,${pad.l - 4},${pad.t + innerH / 2})`}>
        m/s
      </text>
    </svg>
  );
}

/* ─── Page ───────────────────────────────────────────────────────────────────── */

export default function ParkinsonPage() {
  const [profile, setProfile] = useState<MockProfile>("parkinsons");
  const [result, setResult]   = useState<DiagnosisResult | null>(null);
  const [loading, setLoading] = useState(false);

  const runDiagnosis = useCallback(async () => {
    setLoading(true);
    setResult(null);
    await new Promise((r) => setTimeout(r, 1400));
    setResult(DIAGNOSES[profile]);
    setLoading(false);
  }, [profile]);

  const diag   = result ?? DIAGNOSES[profile];
  const color  = STAGE_COLORS[diag.stage];
  const bgTint = STAGE_BG[diag.stage];

  const statusMap = (score: number): { label: string; color: string } => {
    if (score < 0.3) return { label: "정상", color: "#10B981" };
    if (score < 0.6) return { label: "경계", color: "#F59E0B" };
    return { label: "이상", color: "#EF4444" };
  };

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* ── Header ── */}
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-textPri font-semibold text-xl tracking-tight">파킨슨 조기 진단 AI</h1>
            <p className="text-textMuted text-[11px] mt-0.5 font-mono">
              8-biomarker analysis · Hoehn &amp; Yahr staging · UPDRS-aligned indicators
            </p>
          </div>
          <div className="flex items-center gap-3">
            <select
              value={profile}
              onChange={(e) => { setProfile(e.target.value as MockProfile); setResult(null); }}
              className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue"
            >
              {PROFILES.map((p) => <option key={p} value={p}>{PROFILE_LABELS[p]}</option>)}
            </select>
            <button
              onClick={runDiagnosis}
              disabled={loading}
              className="bg-blue hover:bg-blue/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg transition-colors font-mono"
            >
              {loading ? "분석 중…" : "▶  진단 실행"}
            </button>
          </div>
        </header>

        {/* ── Body ── */}
        <div className="flex-1 overflow-y-auto p-6 space-y-5">

          {/* Loading indicator */}
          {loading && (
            <div className="bg-card border border-border rounded-xl p-5 flex items-center gap-4">
              <svg width={24} height={24}>
                <circle cx={12} cy={12} r={9} fill="none" stroke="#3B82F6" strokeWidth={2} strokeDasharray="14 28">
                  <animateTransform attributeName="transform" type="rotate"
                    from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite" />
                </circle>
              </svg>
              <span className="text-textSec text-[13px] font-mono">8개 바이오마커 분석 중… 임상 스테이징 계산 중…</span>
            </div>
          )}

          {/* ── Top row: Stage card + Radar + Biomarkers ── */}
          <div className="grid grid-cols-3 gap-5">

            {/* Stage Card */}
            <div className="rounded-xl border p-5 flex flex-col"
              style={{ borderColor: `${color}40`, background: bgTint }}>
              <div className="text-textMuted text-[11px] font-mono uppercase tracking-wider mb-3">임상 스테이지</div>

              {/* Stage number */}
              <div className="flex items-end gap-3 mb-2">
                <div className="text-[72px] font-black leading-none" style={{ color }}>
                  {diag.stage}
                </div>
                <div className="mb-2">
                  <div className="font-semibold text-[16px] text-textPri">{diag.stageName}</div>
                  <div className="text-textMuted text-[11px] font-mono">{diag.stageEn}</div>
                </div>
              </div>

              {/* Stage bar */}
              <div className="flex gap-1 mb-4">
                {[0, 1, 2, 3].map((s) => (
                  <div key={s} className="h-1.5 flex-1 rounded-full"
                    style={{ background: s <= diag.stage ? STAGE_COLORS[diag.stage] : "#32425B",
                      opacity: s <= diag.stage ? (0.5 + s * 0.15) : 0.3 }} />
                ))}
              </div>

              {/* Confidence */}
              <div className="flex items-center justify-between mb-1">
                <span className="text-textMuted text-[11px] font-mono">신뢰도</span>
                <span className="font-bold text-[13px]" style={{ color }}>
                  {(diag.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-card overflow-hidden mb-4">
                <div className="h-full rounded-full" style={{ width: `${diag.confidence * 100}%`, background: color }} />
              </div>

              {/* PD flag */}
              <div className={`rounded-lg px-3 py-2 text-[12px] font-semibold text-center ${
                diag.isParkinson ? "bg-red/10 text-red border border-red/25" : "bg-green/10 text-green border border-green/25"
              }`}>
                {diag.isParkinson ? "⚠ 파킨슨 패턴 감지" : "✓ 파킨슨 패턴 미감지"}
              </div>

              <p className="text-textSec text-[11px] leading-relaxed mt-4">{diag.summary}</p>

              <div className="mt-auto pt-3 border-t border-border/40">
                <div className="text-textMuted text-[10px] font-mono">참고 기준</div>
                <div className="text-textMuted text-[10px]">H&amp;Y Scale · MDS-UPDRS · Blin et al.</div>
              </div>
            </div>

            {/* Radar Chart */}
            <div className="rounded-xl border border-border bg-card p-4 flex flex-col items-center">
              <div className="text-textMuted text-[11px] font-mono uppercase tracking-wider mb-2 self-start">
                바이오마커 레이더
              </div>
              <RadarChart profile={profile} color={color} />
              <div className="flex items-center gap-4 mt-2 text-[10px] font-mono text-textMuted">
                <div className="flex items-center gap-1.5">
                  <div className="w-4 border-t border-dashed border-green opacity-60" />
                  <span>정상 기준</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-4 border-t-2" style={{ borderColor: color }} />
                  <span>현재 상태</span>
                </div>
              </div>
            </div>

            {/* Biomarker list */}
            <div className="rounded-xl border border-border bg-card p-4 flex flex-col">
              <div className="text-textMuted text-[11px] font-mono uppercase tracking-wider mb-3">
                바이오마커 세부
              </div>
              <div className="space-y-2 flex-1 overflow-y-auto">
                {BIOMARKERS.map((b) => {
                  const val = b.values[profile];
                  const st  = statusMap(val.score);
                  return (
                    <div key={b.id} className="bg-surface rounded-lg px-3 py-2">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-textPri text-[11px] font-semibold">{b.name}</span>
                        <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded"
                          style={{ color: st.color, background: `${st.color}18` }}>
                          {st.label}
                        </span>
                      </div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-textMuted text-[10px] font-mono">{b.nameEn}</span>
                        <span className="text-[11px] font-mono font-bold" style={{ color: st.color }}>
                          {val.display} {b.unit}
                        </span>
                      </div>
                      {/* Mini bar */}
                      <div className="h-1 rounded-full bg-[#1a2438] overflow-hidden">
                        <div className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${val.score * 100}%`, background: st.color, opacity: 0.75 }} />
                      </div>
                      <div className="text-[9px] font-mono text-textMuted mt-0.5">
                        정상: {b.normalRange} {b.unit}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* ── Bottom row: Trend + Recommendations ── */}
          <div className="grid grid-cols-2 gap-5">

            {/* 7-day trend */}
            <div className="rounded-xl border border-border bg-card p-5">
              <div className="flex items-center justify-between mb-3">
                <div className="text-textMuted text-[11px] font-mono uppercase tracking-wider">7일 진행 추이</div>
                <div className="flex items-center gap-4 text-[10px] font-mono text-textMuted">
                  <span className="flex items-center gap-1"><span className="inline-block w-5 border-t-2 border-[#38BDF8]" /> 보행속도</span>
                  <span className="flex items-center gap-1"><span className="inline-block w-5 border-t border-dashed border-[#F87171]" /> 떨림지수</span>
                </div>
              </div>
              <TrendChart profile={profile} />
              {/* Stage legend */}
              <div className="flex items-center gap-3 mt-2">
                {["정상", "전구기", "초기 의심", "중등도"].map((s, i) => (
                  <div key={i} className="flex items-center gap-1 text-[9.5px] font-mono text-textMuted">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: STAGE_COLORS[i] }} />
                    <span>Stage {i}: {s}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            <div className="rounded-xl border border-border bg-card p-5">
              <div className="text-textMuted text-[11px] font-mono uppercase tracking-wider mb-3">임상 권고사항</div>
              <div className="space-y-2 mb-5">
                {diag.recommendations.map((rec, i) => (
                  <div key={i} className="flex items-start gap-2.5 text-[12px]">
                    <span className="mt-0.5 w-4 h-4 rounded-full shrink-0 flex items-center justify-center text-[9px] font-bold"
                      style={{ background: `${color}25`, color }}>
                      {i + 1}
                    </span>
                    <span className="text-textSec">{rec}</span>
                  </div>
                ))}
              </div>

              {/* Stage guide */}
              <div className="border-t border-border/40 pt-3">
                <div className="text-textMuted text-[10px] font-mono uppercase tracking-wider mb-2">H&amp;Y 스테이지 가이드</div>
                <div className="grid grid-cols-2 gap-1.5">
                  {[
                    { s: 0, n: "정상",     d: "이상 없음" },
                    { s: 1, n: "전구기",   d: "전구 증상 관찰" },
                    { s: 2, n: "초기 의심", d: "양측 증상 초기" },
                    { s: 3, n: "중등도",   d: "균형 장애 동반" },
                  ].map(({ s, n, d }) => (
                    <div key={s} className="rounded-lg px-2 py-1.5 border"
                      style={{
                        borderColor: s === diag.stage ? `${STAGE_COLORS[s]}50` : "#32425B",
                        background: s === diag.stage ? `${STAGE_COLORS[s]}12` : "transparent",
                      }}>
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <span className="w-2 h-2 rounded-full" style={{ background: STAGE_COLORS[s] }} />
                        <span className="text-[10px] font-bold" style={{ color: s === diag.stage ? STAGE_COLORS[s] : "#607085" }}>
                          Stage {s} · {n}
                        </span>
                      </div>
                      <div className="text-[9px] text-textMuted">{d}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="bg-amber/5 border border-amber/20 rounded-xl px-5 py-3 text-[11px] text-amber/80 font-mono">
            ⚠ 본 결과는 보행 센서 데이터 기반 참고 지표이며 의학적 진단이 아닙니다.
            최종 진단은 반드시 신경과 전문의의 임상 평가(UPDRS, DaTscan 등)를 통해 이루어져야 합니다.
          </div>
        </div>
      </main>
    </div>
  );
}
