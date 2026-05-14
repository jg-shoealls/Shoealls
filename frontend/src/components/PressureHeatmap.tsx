"use client";

import { useEffect, useState, type ReactElement } from "react";
import type { MockProfile } from "@/lib/mockSensorData";

const ROWS = 16;
const COLS = 8;
type Grid = number[][];

const BASE: Record<MockProfile, (r: number, c: number) => number> = {
  normal: (r, c) => {
    const heel = r > 11 ? 0.85 : 0;
    const forefoot = r < 4 ? 0.75 : 0;
    const arch = r >= 4 && r <= 11 ? 0.15 : 0;
    const lateral = c >= 2 && c <= 5 ? 0.1 : 0;
    return heel + forefoot + arch + lateral;
  },
  parkinsons: (r, c) => {
    const forefoot = r < 6 ? 0.9 : 0;
    const heel = r > 13 ? 0.3 : 0;
    const medial = c >= 3 && c <= 5 ? 0.15 : 0;
    return forefoot + heel + medial;
  },
  stroke: (r, c) => {
    const side = c >= 4 ? 0.9 : 0.3;
    const heel = r > 10 ? 0.5 : 0;
    const toe = r < 3 ? 0.4 : 0;
    return (side + heel + toe) * 0.6;
  },
  fall_risk: (r, c) => {
    const lateral = c < 2 || c > 5 ? 0.75 : 0.2;
    const toe = r < 5 ? 0.6 : 0;
    const heel = r > 12 ? 0.5 : 0;
    return lateral + toe + heel;
  },
};

function makeGrid(profile: MockProfile, phase: number): Grid {
  return Array.from({ length: ROWS }, (_, r) =>
    Array.from({ length: COLS }, (_, c) => {
      const base = BASE[profile](r, c);
      const wave = Math.sin(phase + r * 0.4) * 0.12;
      return Math.max(0, Math.min(1, base + wave + (Math.random() - 0.5) * 0.06));
    })
  );
}

function downsample(grid: Grid): number[][] {
  return Array.from({ length: 8 }, (_, r) =>
    Array.from({ length: 4 }, (_, c) => {
      const vals = [
        grid[r * 2]?.[c * 2] ?? 0,
        grid[r * 2]?.[c * 2 + 1] ?? 0,
        grid[r * 2 + 1]?.[c * 2] ?? 0,
        grid[r * 2 + 1]?.[c * 2 + 1] ?? 0,
      ];
      return vals.reduce((a, b) => a + b, 0) / 4;
    })
  );
}

function valToHSL(v: number): string {
  const hue = (1 - v) * 240;
  return `hsl(${hue.toFixed(0)}, 90%, 55%)`;
}

function contourColor(v: number): string {
  if (v < 0.2) return "#1e3a5f";
  if (v < 0.4) return "#1d4ed8";
  if (v < 0.6) return "#10b981";
  if (v < 0.8) return "#f59e0b";
  return "#ef4444";
}

// ── Isometric 3D chart ──────────────────────────────────────────
function IsoChart({ grid }: { grid: Grid }) {
  const small = downsample(grid);
  const RS = 8, CS = 4;
  const CW = 18, MAX_H = 40;
  const IX = CW / 2, IY = CW / 4;
  const W = (CS + RS) * IX + 10;
  const H = (CS + RS) * IY + MAX_H + 20;

  const columns: ReactElement[] = [];
  for (let r = RS - 1; r >= 0; r--) {
    for (let c = 0; c < CS; c++) {
      const v = small[r][c];
      const h = Math.max(2, v * MAX_H);
      const sx = (c - r) * IX + W / 2 - IX;
      const sy = (c + r) * IY + 10;
      const hue = ((1 - v) * 240).toFixed(0);
      const top  = `hsl(${hue},90%,55%)`;
      const mid  = `hsl(${hue},80%,45%)`;
      const dark = `hsl(${hue},70%,30%)`;
      const pts = (pts: [number, number][]) => pts.map((p) => p.join(",")).join(" ");
      columns.push(
        <g key={`${r}-${c}`}>
          <polygon points={pts([[sx - IX, sy - h + IY],[sx, sy - h + IY * 2],[sx, sy + IY * 2],[sx - IX, sy + IY]])} fill={dark} />
          <polygon points={pts([[sx + IX, sy - h + IY],[sx, sy - h + IY * 2],[sx, sy + IY * 2],[sx + IX, sy + IY]])} fill={mid} />
          <polygon points={pts([[sx, sy - h],[sx + IX, sy - h + IY],[sx, sy - h + IY * 2],[sx - IX, sy - h + IY]])} fill={top} />
        </g>
      );
    }
  }
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
      {columns}
    </svg>
  );
}

// ── Foot shape (right foot, plantar view, toes at top) ──────────
// viewBox: 0 0 172 278  —  rendered at 148×239 (86%)

// Main foot body: smooth bezier, anatomical proportions
// Medial arch on left, wide heel at bottom, ball widest at ~y=80
const FOOT_BODY =
  "M 40 78 " +
  // medial side up toward big toe
  "C 35 62, 31 46, 37 32 " +
  "C 42 20, 56 15, 66 21 " +
  // top of foot (concave dip between toe bases)
  "C 73 16, 83 11, 90 12 " +
  "C 97 11, 107 14, 115 20 " +
  "C 121 24, 128 32, 134 42 " +
  // lateral ball
  "C 140 54, 144 68, 146 84 " +
  // lateral side down
  "C 150 100, 151 120, 149 142 " +
  "C 147 164, 143 186, 137 208 " +
  "C 129 230, 117 250, 101 260 " +
  // heel bottom
  "C 91 268, 77 268, 64 261 " +
  // medial heel
  "C 51 254, 40 240, 36 224 " +
  // medial arch (concave — the arch!)
  "C 32 208, 31 191, 32 174 " +
  "C 33 157, 34 140, 35 122 " +
  "C 36 105, 37 91, 40 78 Z";

// Toe ellipses: big toe largest, cascade to little toe
const TOES = [
  { cx: 50,  cy: 26, rx: 20, ry: 24 }, // hallux (big)
  { cx: 76,  cy: 16, rx: 13, ry: 17 }, // 2nd
  { cx: 96,  cy: 15, rx: 11, ry: 15 }, // 3rd
  { cx: 113, cy: 22, rx: 10, ry: 13 }, // 4th
  { cx: 127, cy: 36, rx: 8,  ry: 11 }, // 5th (little)
];

// Pressure grid: row 0 = forefoot/toes, row 15 = heel
const GX0 = 38, GW = 12.5;   // col 0..7 → x 38..125
const GY0 = 4,  GH = 17;     // row 0..15 → y 4..272

function FootHeatmap({ grid }: { grid: Grid }) {
  const cells: ReactElement[] = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const v = grid[r][c];
      cells.push(
        <rect
          key={`${r}-${c}`}
          x={GX0 + c * GW}
          y={GY0 + r * GH}
          width={GW + 0.8}
          height={GH + 0.8}
          fill={contourColor(v)}
        />
      );
    }
  }

  return (
    // viewBox 172×278, rendered 148×239
    <svg width={148} height={239} viewBox="0 0 172 278" style={{ display: "block" }}>
      <defs>
        {/* Clip path = body + all 5 toes */}
        <clipPath id="foot-clip">
          <path d={FOOT_BODY} />
          {TOES.map((t, i) => (
            <ellipse key={i} cx={t.cx} cy={t.cy} rx={t.rx} ry={t.ry} />
          ))}
        </clipPath>

        {/* Subtle inner glow filter */}
        <filter id="foot-glow" x="-8%" y="-4%" width="116%" height="108%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Pressure data clipped to foot shape */}
      <g clipPath="url(#foot-clip)" filter="url(#foot-glow)">
        {cells}
      </g>

      {/* Dark edge vignette on foot (visual polish) */}
      <path
        d={FOOT_BODY}
        fill="none"
        stroke="rgba(0,0,0,0.35)"
        strokeWidth={6}
        clipPath="url(#foot-clip)"
      />

      {/* Foot body outline */}
      <path
        d={FOOT_BODY}
        fill="none"
        stroke="rgba(148,163,184,0.55)"
        strokeWidth={1.4}
        strokeLinejoin="round"
      />

      {/* Toe outlines */}
      {TOES.map((t, i) => (
        <ellipse
          key={i}
          cx={t.cx} cy={t.cy} rx={t.rx} ry={t.ry}
          fill="none"
          stroke="rgba(148,163,184,0.45)"
          strokeWidth={1}
        />
      ))}

      {/* Toenail highlights — small bright ellipse near top of each toe */}
      {TOES.map((t, i) => (
        <ellipse
          key={`nl-${i}`}
          cx={t.cx}
          cy={t.cy - t.ry * 0.38}
          rx={t.rx * 0.46}
          ry={t.ry * 0.26}
          fill="rgba(255,255,255,0.11)"
          stroke="rgba(255,255,255,0.22)"
          strokeWidth={0.6}
        />
      ))}

      {/* Medial arch indicator line (subtle anatomical landmark) */}
      <path
        d="M 34 174 C 33 158, 34 141, 35 122"
        fill="none"
        stroke="rgba(148,163,184,0.2)"
        strokeWidth={2}
        strokeDasharray="3 3"
      />
    </svg>
  );
}

// ── Labels & legend ─────────────────────────────────────────────
const LABELS: Record<MockProfile, string> = {
  normal:    "균형 분포",
  parkinsons:"전족부 과부하 (발 끌림)",
  stroke:    "편측 비대칭 하중",
  fall_risk: "외측 불안정",
};

const LEGEND = [
  { color: "#ef4444", label: "80–100%" },
  { color: "#f59e0b", label: "60–80%" },
  { color: "#10b981", label: "40–60%" },
  { color: "#1d4ed8", label: "20–40%" },
  { color: "#1e3a5f", label: "0–20%"  },
];

// ── Main component ───────────────────────────────────────────────
export default function PressureHeatmap({ profile }: { profile: MockProfile }) {
  const [grid, setGrid] = useState<Grid>(() => makeGrid(profile, 0));
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setPhase((p) => {
        const next = p + 0.25;
        setGrid(makeGrid(profile, next));
        return next;
      });
    }, 160);
    return () => clearInterval(id);
  }, [profile]);

  return (
    <div className="bg-card rounded-xl border border-border p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-textPri font-semibold text-[13px]">족저압 분포 히트맵</span>
        <span className="text-[11px] text-amber font-medium">{LABELS[profile]}</span>
      </div>

      <div className="flex gap-3 items-start">
        {/* 3D Isometric */}
        <div className="flex flex-col items-center gap-1 shrink-0">
          <span className="text-textMuted text-[10px]">3D 등각투영</span>
          <div className="overflow-hidden" style={{ width: 160, height: 160 }}>
            <IsoChart grid={grid} />
          </div>
        </div>

        {/* Foot heatmap */}
        <div className="flex flex-col items-center gap-1 shrink-0">
          <span className="text-textMuted text-[10px]">족저압 등고선</span>
          <FootHeatmap grid={grid} />
        </div>

        {/* Legend */}
        <div className="flex flex-col gap-1.5 mt-6 ml-auto">
          {LEGEND.map((l) => (
            <div key={l.label} className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm shrink-0" style={{ background: l.color }} />
              <span className="text-textMuted text-[9px] tabular-nums">{l.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
