"use client";

import { useCallback, useMemo, useState } from "react";
import Sidebar from "@/components/Sidebar";
import { MockProfile, PROFILE_LABELS } from "@/lib/mockSensorData";

/* ─── Constants ─────────────────────────────────────────────────────────────── */

const PROFILES: MockProfile[] = ["normal", "parkinsons", "stroke", "fall_risk"];

const NW = 174;   // node width
const NH = 84;    // node height
const NR = 8;     // corner radius
const COL_GAP = 44; // gap between layers

type Status = "idle" | "loading" | "done";

/* ─── Data definitions ───────────────────────────────────────────────────────── */

interface GNode {
  id: string; label: string; sub: string;
  x: number; y: number; color: string; layer: number;
}

const COLS = [28, 28 + NW + COL_GAP, 28 + (NW + COL_GAP) * 2, 28 + (NW + COL_GAP) * 3, 28 + (NW + COL_GAP) * 4];
// COLS = [28, 246, 464, 682, 900]

const GNODES: GNode[] = [
  { id: "imu",       label: "IMU 센서",        sub: "[128,6] · 6-axis · 128 Hz", x: COLS[0], y: 68,  color: "#38BDF8", layer: 0 },
  { id: "pressure",  label: "족저압 센서",      sub: "[16,8] · 양발 매트릭스",    x: COLS[0], y: 216, color: "#34D399", layer: 0 },
  { id: "skeleton",  label: "스켈레톤",          sub: "[128,17,3] · 관절 궤적",    x: COLS[0], y: 364, color: "#C084FC", layer: 0 },
  { id: "imu_enc",   label: "IMU Encoder",      sub: "1D-CNN · BiLSTM",           x: COLS[1], y: 68,  color: "#38BDF8", layer: 1 },
  { id: "press_enc", label: "Pressure Enc.",    sub: "2D-CNN · Spatial",          x: COLS[1], y: 216, color: "#34D399", layer: 1 },
  { id: "skel_enc",  label: "Skeleton Enc.",    sub: "Graph Conv Net",            x: COLS[1], y: 364, color: "#C084FC", layer: 1 },
  { id: "attention", label: "Cross-Modal Attn", sub: "Transformer · Fusion",      x: COLS[2], y: 216, color: "#FBBF24", layer: 2 },
  { id: "classify",  label: "보행 분류",         sub: "4-class · BiLSTM",          x: COLS[3], y: 88,  color: "#38BDF8", layer: 3 },
  { id: "disease",   label: "위험 징후",         sub: "11 biomarkers",             x: COLS[3], y: 216, color: "#F87171", layer: 3 },
  { id: "injury",    label: "낙상/부상",         sub: "6 risk types",              x: COLS[3], y: 344, color: "#FBBF24", layer: 3 },
  { id: "report",    label: "종합 리포트",       sub: "Chain-of-Reasoning",        x: COLS[4], y: 216, color: "#34D399", layer: 4 },
];

const NODE_MAP = Object.fromEntries(GNODES.map((n) => [n.id, n]));

const EDGES: [string, string][] = [
  ["imu", "imu_enc"], ["pressure", "press_enc"], ["skeleton", "skel_enc"],
  ["imu_enc", "attention"], ["press_enc", "attention"], ["skel_enc", "attention"],
  ["attention", "classify"], ["attention", "disease"], ["attention", "injury"],
  ["classify", "report"], ["disease", "report"], ["injury", "report"],
];

const TIMELINE: { delay: number; doneAt: number; ids: string[] }[] = [
  { delay: 0,    doneAt: 720,  ids: ["imu", "pressure", "skeleton"] },
  { delay: 900,  doneAt: 1680, ids: ["imu_enc", "press_enc", "skel_enc"] },
  { delay: 1860, doneAt: 2740, ids: ["attention"] },
  { delay: 2920, doneAt: 3720, ids: ["classify", "disease", "injury"] },
  { delay: 3900, doneAt: 4580, ids: ["report"] },
];

const METRICS: Record<string, Record<MockProfile, string>> = {
  imu:       { normal: "안정 파형",      parkinsons: "4-6 Hz 떨림",    stroke: "편측 비대칭",   fall_risk: "가속 불안정" },
  pressure:  { normal: "균형 분포",      parkinsons: "전족부 과부하",   stroke: "좌우 70:30",    fall_risk: "외측 불균형" },
  skeleton:  { normal: "정상 궤적",      parkinsons: "소보 패턴",       stroke: "편마비 궤적",   fall_risk: "균형 이탈" },
  imu_enc:   { normal: "Embed OK",      parkinsons: "떨림 특성 추출",  stroke: "비대칭 임베딩", fall_risk: "불안정 패턴" },
  press_enc: { normal: "Embed OK",      parkinsons: "발끌림 특성",     stroke: "편측 하중",     fall_risk: "외측 로딩" },
  skel_enc:  { normal: "Graph OK",      parkinsons: "소보 그래프",     stroke: "편측 그래프",   fall_risk: "불균형 그래프" },
  attention: { normal: "Fusion OK",     parkinsons: "파킨슨 강조",     stroke: "편마비 패턴",   fall_risk: "위험 감지" },
  classify:  { normal: "안정 96%",      parkinsons: "파킨슨 87%",      stroke: "편마비 91%",    fall_risk: "불안정 82%" },
  disease:   { normal: "이상 없음 97%", parkinsons: "파킨슨 신호 89%", stroke: "뇌졸중 91%",    fall_risk: "낙상 위험 84%" },
  injury:    { normal: "LOW · 94%",     parkinsons: "MED · 76%",       stroke: "HIGH · 91%",    fall_risk: "HIGH · 88%" },
  report:    { normal: "정상 보행",      parkinsons: "신경과 권고",     stroke: "재활 필요",     fall_risk: "즉시 개입" },
};

const CONFIDENCE: Record<string, Record<MockProfile, number>> = {
  classify:  { normal: 0.96, parkinsons: 0.87, stroke: 0.91, fall_risk: 0.82 },
  disease:   { normal: 0.97, parkinsons: 0.89, stroke: 0.91, fall_risk: 0.84 },
  injury:    { normal: 0.94, parkinsons: 0.76, stroke: 0.91, fall_risk: 0.88 },
  report:    { normal: 0.97, parkinsons: 0.87, stroke: 0.91, fall_risk: 0.88 },
};

const NODE_DESC: Record<string, { arch: string; detail: string }> = {
  imu:       { arch: "6-axis Inertial Measurement Unit", detail: "128fps 가속도 3축 + 자이로 3축. 보행 동작 시계열 캡처." },
  pressure:  { arch: "Plantar Pressure Matrix", detail: "16×8 압력 센서. 좌우 발 족저압 실시간 공간 분포 측정." },
  skeleton:  { arch: "Skeleton Keypoint Stream", detail: "17개 관절점 × 3D 좌표 × 128 프레임 시퀀스." },
  imu_enc:   { arch: "1D-CNN + Bidirectional LSTM", detail: "CNN 지역 특성 → BiLSTM 장기 시퀀스 패턴 학습." },
  press_enc: { arch: "2D Convolutional Encoder", detail: "족저압 공간 패턴 2D-CNN 인코딩. 체중 전이 특성 추출." },
  skel_enc:  { arch: "Graph Convolutional Network", detail: "관절 간 연결 관계를 그래프로 모델링하여 움직임 패턴 학습." },
  attention: { arch: "Cross-Modal Transformer Attention", detail: "세 인코더 출력을 교차 어텐션으로 융합. 모달 간 상호 정보 학습." },
  classify:  { arch: "4-Class Gait Classifier", detail: "정상 / 파킨슨 / 편마비 / 불안정. BiLSTM + Softmax." },
  disease:   { arch: "Disease Risk Monitor", detail: "11개 바이오마커 기반 위험 징후 관찰. ML 분류 + 임계값 보정." },
  injury:    { arch: "Injury Risk Evaluator", detail: "6가지 부상 유형 복합 위험도. 족저압 + IMU 앙상블 기반." },
  report:    { arch: "Chain-of-Reasoning Engine", detail: "4단계 추론 트레이스: 이상 감지 → 근거 수집 → 감별 → 판정." },
};

const LAYER_LABELS = ["Input", "Encoder", "Fusion", "Output", "Report"];

const CANVAS_W = COLS[4] + NW + 28;  // 1102
const CANVAS_H = 364 + NH + 28;      // 476

/* ─── Helpers ────────────────────────────────────────────────────────────────── */

function trunc(s: string, n: number) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function bezier(from: GNode, to: GNode) {
  const sx = from.x + NW, sy = from.y + NH / 2;
  const ex = to.x,         ey = to.y + NH / 2;
  const cp = (ex - sx) * 0.5;
  return `M${sx},${sy} C${sx + cp},${sy} ${ex - cp},${ey} ${ex},${ey}`;
}

function barSeed(id: string, profile: MockProfile): number[] {
  const base = (id.charCodeAt(0) + profile.charCodeAt(0)) % 97;
  return Array.from({ length: 10 }, (_, i) => ((base * (i * 7 + 3)) % 14) + 3);
}

function initStatus(): Record<string, Status> {
  return Object.fromEntries(GNODES.map((n) => [n.id, "idle" as Status]));
}

/* ─── SVG Node Glyph ─────────────────────────────────────────────────────────── */

function NodeGlyph({ node, s, metric, conf, bars, selected, onClick }: {
  node: GNode; s: Status; metric: string | null; conf: number | null;
  bars: number[]; selected: boolean; onClick: () => void;
}) {
  const isDone = s === "done";
  const isLoad = s === "loading";
  const isIdle = s === "idle";

  const bg     = isDone ? `${node.color}12` : isLoad ? `${node.color}0A` : "#1a2438";
  const stroke = isDone ? node.color : isLoad ? node.color : selected ? node.color : "#2d3f58";
  const sOp    = isDone ? 0.5 : isLoad ? 0.35 : selected ? 0.5 : 0.5;
  const gOp    = isIdle && !selected ? 0.38 : 1;

  // Confidence bar width (max 150px)
  const confW = conf != null ? Math.round(conf * 150) : 0;

  return (
    <g transform={`translate(${node.x},${node.y})`} onClick={onClick}
      style={{ cursor: "pointer" }} opacity={gOp}>

      {/* Done outer glow */}
      {isDone && (
        <rect x={-6} y={-6} width={NW + 12} height={NH + 12}
          rx={NR + 4} fill={node.color} opacity={0.06} />
      )}

      {/* Selection ring */}
      {selected && (
        <rect x={-3} y={-3} width={NW + 6} height={NH + 6}
          rx={NR + 2} fill="none" stroke={node.color} strokeWidth={2} strokeOpacity={0.65} />
      )}

      {/* Body */}
      <rect width={NW} height={NH} rx={NR} fill={bg}
        stroke={stroke} strokeWidth={1.5} strokeOpacity={sOp}>
        {isLoad && (
          <animate attributeName="stroke-opacity" values="0.12;0.55;0.12" dur="1s" repeatCount="indefinite" />
        )}
      </rect>

      {/* Colored header accent (top 3px) */}
      <rect width={NW} height={3} rx={NR}
        fill={node.color} opacity={isDone ? 0.8 : isLoad ? 0.5 : 0.2}>
        {isLoad && (
          <animate attributeName="opacity" values="0.25;0.6;0.25" dur="1s" repeatCount="indefinite" />
        )}
      </rect>

      {/* Status dot */}
      <circle cx={14} cy={22} r={4.5} fill={isDone || isLoad ? node.color : "#2d3f58"}>
        {isLoad && (
          <animate attributeName="opacity" values="0.4;1;0.4" dur="1s" repeatCount="indefinite" />
        )}
      </circle>

      {/* Done ✓ */}
      {isDone && (
        <text x={NW - 14} y={26} fontSize={11} fontWeight={700} fill="#34D399" textAnchor="middle">✓</text>
      )}

      {/* Loading ring */}
      {isLoad && (
        <circle cx={NW - 14} cy={22} r={5} fill="none" stroke={node.color} strokeWidth={1.5} strokeOpacity={0.6}
          strokeDasharray="8 6">
          <animateTransform attributeName="transform" type="rotate"
            from={`0 ${NW - 14} 22`} to={`360 ${NW - 14} 22`}
            dur="1.2s" repeatCount="indefinite" />
        </circle>
      )}

      {/* Label */}
      <text x={27} y={26} fontSize={12.5} fontWeight={600} fill="#E2E8F0"
        fontFamily="ui-sans-serif, system-ui, sans-serif">
        {trunc(node.label, 15)}
      </text>

      {/* Sub (monospace) */}
      <text x={12} y={45} fontSize={9.5} fill="#4a6080"
        fontFamily="'JetBrains Mono', 'Fira Code', ui-monospace, monospace">
        {trunc(node.sub, 27)}
      </text>

      {/* Metric */}
      {isDone && metric ? (
        <text x={12} y={62} fontSize={11} fontWeight={600} fill={node.color}
          fontFamily="ui-sans-serif, system-ui, sans-serif">
          {trunc(metric, 22)}
        </text>
      ) : isLoad ? (
        <text x={12} y={62} fontSize={10} fill="#4a6080" fontFamily="ui-sans-serif">
          processing…
          <animate attributeName="opacity" values="0.3;1;0.3" dur="1s" repeatCount="indefinite" />
        </text>
      ) : (
        <text x={12} y={62} fontSize={10} fill="#2d3f58" fontFamily="ui-sans-serif">standby</text>
      )}

      {/* Confidence bar (output + report nodes only) */}
      {isDone && conf != null ? (
        <g transform="translate(12, 70)">
          <rect width={150} height={4} rx={2} fill="#1a2438" />
          <rect width={confW} height={4} rx={2} fill={node.color} opacity={0.75} />
        </g>
      ) : isDone && bars.length > 0 ? (
        /* Mini bar chart for non-confidence nodes */
        <g transform="translate(12, 68)">
          {bars.slice(0, 10).map((h, i) => (
            <rect key={i} x={i * 15} y={14 - h} width={10} height={h}
              rx={2} fill={node.color} opacity={0.35 + (h / 14) * 0.4} />
          ))}
        </g>
      ) : null}
    </g>
  );
}

/* ─── Page ───────────────────────────────────────────────────────────────────── */

export default function GraphPage() {
  const [profile, setProfile] = useState<MockProfile>("parkinsons");
  const [status,  setStatus]  = useState<Record<string, Status>>(initStatus);
  const [running, setRunning] = useState(false);
  const [complete, setComplete] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  const changeProfile = (p: MockProfile) => {
    setProfile(p);
    setStatus(initStatus());
    setComplete(false);
    setSelected(null);
  };

  const runPipeline = useCallback(() => {
    setRunning(true);
    setComplete(false);
    setSelected(null);
    setStatus(initStatus());

    const hs: ReturnType<typeof setTimeout>[] = [];
    TIMELINE.forEach(({ delay, doneAt, ids }) => {
      hs.push(
        setTimeout(() => setStatus(p => { const n = { ...p }; ids.forEach(id => { n[id] = "loading"; }); return n; }), delay),
        setTimeout(() => setStatus(p => { const n = { ...p }; ids.forEach(id => { n[id] = "done"; }); return n; }), doneAt),
      );
    });
    hs.push(setTimeout(() => { setRunning(false); setComplete(true); }, TIMELINE.at(-1)!.doneAt + 200));
    return () => hs.forEach(clearTimeout);
  }, []);

  const selNode = selected ? NODE_MAP[selected] : null;

  // Pre-compute bars per node+profile (stable)
  const barsMap = useMemo(() =>
    Object.fromEntries(GNODES.map(n => [n.id, barSeed(n.id, profile)])),
  [profile]);

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden">

        {/* ── Header ── */}
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-textPri font-semibold text-xl tracking-tight">
              AI Pipeline · Vector Node Graph
            </h1>
            <p className="text-textMuted text-[11px] mt-0.5 font-mono">
              SVG vector rendering · multimodal fusion · 5-layer pipeline
            </p>
          </div>
          <div className="flex items-center gap-3">
            <select
              value={profile}
              onChange={(e) => changeProfile(e.target.value as MockProfile)}
              className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue"
            >
              {PROFILES.map((p) => <option key={p} value={p}>{PROFILE_LABELS[p]}</option>)}
            </select>
            {complete && (
              <span className="px-3 py-0.5 rounded-full text-[11px] font-mono font-semibold bg-green/15 text-green border border-green/25">
                ● pipeline complete
              </span>
            )}
            <button
              onClick={runPipeline}
              disabled={running}
              className="bg-blue hover:bg-blue/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg transition-colors font-mono tracking-wide"
            >
              {running ? "running…" : "▶  Run Pipeline"}
            </button>
          </div>
        </header>

        {/* ── Body ── */}
        <div className="flex-1 flex overflow-hidden">

          {/* Graph area */}
          <div className="flex-1 overflow-auto p-5 bg-[#0A1220]">
            <svg
              width={CANVAS_W} height={CANVAS_H}
              viewBox={`0 0 ${CANVAS_W} ${CANVAS_H}`}
              style={{ minWidth: CANVAS_W, display: "block" }}
            >
              <defs>
                {/* Dot grid pattern */}
                <pattern id="dotgrid" width={24} height={24} patternUnits="userSpaceOnUse">
                  <circle cx={1} cy={1} r={0.8} fill="#1E2940" />
                </pattern>

                {/* Glow filter for edges */}
                <filter id="glow-edge" x="-40%" y="-40%" width="180%" height="180%">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>

                {/* Subtle glow for node halos */}
                <filter id="glow-node" x="-25%" y="-25%" width="150%" height="150%">
                  <feGaussianBlur stdDeviation="6" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>

              {/* Background dot grid */}
              <rect width={CANVAS_W} height={CANVAS_H} fill="url(#dotgrid)" />

              {/* Layer column bands */}
              {COLS.map((cx, li) => (
                <g key={`col-${li}`}>
                  {/* Subtle column highlight */}
                  <rect x={cx - 8} y={0} width={NW + 16} height={CANVAS_H}
                    fill={li === 2 ? "#FBBF2406" : "#38BDF806"} rx={4} />

                  {/* Layer header pill */}
                  <rect x={cx + NW / 2 - 32} y={8} width={64} height={18}
                    rx={9} fill="#111c30" stroke="#2d3f58" strokeWidth={0.8} />
                  <text x={cx + NW / 2} y={20.5}
                    textAnchor="middle" fontSize={9.5} fontWeight={600}
                    fill="#4a6080" letterSpacing={0.8}
                    fontFamily="'JetBrains Mono', ui-monospace, monospace">
                    {LAYER_LABELS[li]}
                  </text>
                </g>
              ))}

              {/* ── Edges ── */}
              {EDGES.map(([f, t]) => {
                const from = NODE_MAP[f];
                const to   = NODE_MAP[t];
                const path = bezier(from, to);
                const active = status[f] === "done";

                return (
                  <g key={`e-${f}-${t}`}>
                    {/* Glow behind active edge */}
                    {active && (
                      <path d={path} stroke={from.color} strokeWidth={5}
                        fill="none" strokeOpacity={0.1} filter="url(#glow-edge)" />
                    )}
                    {/* Main edge */}
                    <path d={path}
                      stroke={active ? from.color : "#1e2f45"}
                      strokeWidth={active ? 1.5 : 1}
                      fill="none"
                      strokeOpacity={active ? 0.8 : 0.5}
                      strokeDasharray={active ? undefined : "4,6"} />
                    {/* Flow dots */}
                    {active && (
                      <>
                        <circle r={5} fill={from.color} opacity={0.95}>
                          <animateMotion dur="1.55s" repeatCount="indefinite" path={path} />
                        </circle>
                        <circle r={3.5} fill={from.color} opacity={0.5}>
                          <animateMotion dur="1.55s" repeatCount="indefinite" begin="0.52s" path={path} />
                        </circle>
                        <circle r={2} fill={from.color} opacity={0.28}>
                          <animateMotion dur="1.55s" repeatCount="indefinite" begin="1.04s" path={path} />
                        </circle>
                      </>
                    )}
                  </g>
                );
              })}

              {/* ── Nodes ── */}
              {GNODES.map((node) => {
                const s = status[node.id];
                const isDone = s === "done";
                return (
                  <NodeGlyph
                    key={node.id}
                    node={node}
                    s={s}
                    metric={isDone ? (METRICS[node.id]?.[profile] ?? null) : null}
                    conf={isDone ? (CONFIDENCE[node.id]?.[profile] ?? null) : null}
                    bars={barsMap[node.id] ?? []}
                    selected={selected === node.id}
                    onClick={() => setSelected(selected === node.id ? null : node.id)}
                  />
                );
              })}

              {/* ── Legend ── */}
              <g transform={`translate(28,${CANVAS_H - 20})`}
                fontFamily="'JetBrains Mono', ui-monospace, monospace" fontSize={9.5} fill="#4a6080">
                <circle cx={5} cy={5} r={4} fill="#2d3f58" opacity={0.6} />
                <text x={14} y={9}>idle</text>
                <circle cx={62} cy={5} r={4} fill="#38BDF8">
                  <animate attributeName="opacity" values="0.3;1;0.3" dur="1s" repeatCount="indefinite" />
                </circle>
                <text x={71} y={9}>loading</text>
                <circle cx={136} cy={5} r={4} fill="#34D399" />
                <text x={145} y={9}>done</text>
                <line x1={200} y1={5} x2={232} y2={5} stroke="#38BDF8" strokeWidth={1.5} />
                <circle r={3} fill="#38BDF8" opacity={0.9}>
                  <animateMotion dur="1.55s" repeatCount="indefinite" path="M200,5 L232,5" />
                </circle>
                <text x={238} y={9}>data flow</text>
              </g>
            </svg>
          </div>

          {/* ── Detail Panel ── */}
          <aside
            className="border-l border-border bg-surface overflow-y-auto shrink-0 transition-all duration-200"
            style={{ width: selNode ? 268 : 0, opacity: selNode ? 1 : 0 }}
          >
            {selNode && (() => {
              const s    = status[selNode.id];
              const desc = NODE_DESC[selNode.id];
              const metric = s === "done" ? METRICS[selNode.id]?.[profile] : null;
              const conf   = s === "done" ? CONFIDENCE[selNode.id]?.[profile] : null;
              return (
                <div className="p-5 w-[268px]">
                  {/* Title */}
                  <div className="flex items-center justify-between mb-5">
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full" style={{ background: selNode.color }} />
                      <span className="text-textPri font-semibold text-[14px]">{selNode.label}</span>
                    </div>
                    <button onClick={() => setSelected(null)}
                      className="text-textMuted hover:text-textPri text-lg leading-none">×</button>
                  </div>

                  <div className="space-y-3 text-[12px]">
                    {/* Layer */}
                    <div className="bg-card rounded-lg p-3">
                      <div className="text-textMuted text-[10px] font-mono uppercase tracking-wider mb-1">Layer</div>
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold"
                          style={{ background: `${selNode.color}22`, color: selNode.color }}>
                          L{selNode.layer}
                        </span>
                        <span className="text-textPri">{LAYER_LABELS[selNode.layer]}</span>
                      </div>
                    </div>

                    {/* Architecture */}
                    <div className="bg-card rounded-lg p-3">
                      <div className="text-textMuted text-[10px] font-mono uppercase tracking-wider mb-1">Architecture</div>
                      <div className="text-textPri font-mono text-[11px]">{desc.arch}</div>
                    </div>

                    {/* Description */}
                    <div className="bg-card rounded-lg p-3">
                      <div className="text-textMuted text-[10px] font-mono uppercase tracking-wider mb-1">Description</div>
                      <div className="text-textSec leading-relaxed">{desc.detail}</div>
                    </div>

                    {/* Result */}
                    {s === "done" && metric && (
                      <div className="rounded-lg p-3 border"
                        style={{ borderColor: `${selNode.color}40`, background: `${selNode.color}0E` }}>
                        <div className="text-textMuted text-[10px] font-mono uppercase tracking-wider mb-1.5">
                          Result — {PROFILE_LABELS[profile]}
                        </div>
                        <div className="font-semibold text-[13px]" style={{ color: selNode.color }}>
                          {metric}
                        </div>
                        {conf != null && (
                          <div className="mt-2">
                            <div className="h-1.5 rounded-full bg-card overflow-hidden">
                              <div className="h-full rounded-full transition-all duration-500"
                                style={{ width: `${conf * 100}%`, background: selNode.color, opacity: 0.75 }} />
                            </div>
                            <div className="text-textMuted text-[10px] font-mono mt-1">
                              confidence {(conf * 100).toFixed(0)}%
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Status */}
                    <div className="bg-card rounded-lg p-3">
                      <div className="text-textMuted text-[10px] font-mono uppercase tracking-wider mb-1">Status</div>
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full"
                          style={{ background: s === "done" ? "#34D399" : s === "loading" ? selNode.color : "#2d3f58" }} />
                        <span className="text-textPri font-mono text-[11px]">{s}</span>
                      </div>
                    </div>

                    {/* Connections */}
                    <div className="bg-card rounded-lg p-3">
                      <div className="text-textMuted text-[10px] font-mono uppercase tracking-wider mb-2">Connections</div>
                      <div className="space-y-1.5">
                        {EDGES.filter(([, t]) => t === selected).map(([f]) => (
                          <div key={`in-${f}`} className="flex items-center gap-2 text-[11px] font-mono">
                            <span style={{ color: NODE_MAP[f].color }}>→</span>
                            <span className="text-textMuted">from</span>
                            <span className="text-textSec">{NODE_MAP[f].label}</span>
                          </div>
                        ))}
                        {EDGES.filter(([f]) => f === selected).map(([, t]) => (
                          <div key={`out-${t}`} className="flex items-center gap-2 text-[11px] font-mono">
                            <span style={{ color: NODE_MAP[t].color }}>→</span>
                            <span className="text-textMuted">to</span>
                            <span className="text-textSec">{NODE_MAP[t].label}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()}
          </aside>
        </div>
      </main>
    </div>
  );
}
