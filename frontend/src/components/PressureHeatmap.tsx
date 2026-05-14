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

// Downsample 16×8 → 8×4 for isometric view
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

// Isometric 3D column chart (left panel)
function IsoChart({ grid }: { grid: Grid }) {
  const small = downsample(grid);
  const ROWS_S = 8;
  const COLS_S = 4;
  const CW = 18; // cell width
  const MAX_H = 40;
  const ISO_X = CW / 2;
  const ISO_Y = CW / 4;
  const W = (COLS_S + ROWS_S) * ISO_X + 10;
  const H = (COLS_S + ROWS_S) * ISO_Y + MAX_H + 20;

  const columns: ReactElement[] = [];
  // render back-to-front: row desc, col asc
  for (let r = ROWS_S - 1; r >= 0; r--) {
    for (let c = 0; c < COLS_S; c++) {
      const v = small[r][c];
      const h = Math.max(2, v * MAX_H);
      // isometric screen position
      const sx = (c - r) * ISO_X + W / 2 - ISO_X;
      const sy = (c + r) * ISO_Y + 10;
      const color = valToHSL(v);
      const dark = `hsl(${((1 - v) * 240).toFixed(0)}, 70%, 35%)`;
      const mid = `hsl(${((1 - v) * 240).toFixed(0)}, 80%, 45%)`;
      // top face
      const top = [
        [sx, sy - h],
        [sx + ISO_X, sy - h + ISO_Y],
        [sx, sy - h + ISO_Y * 2],
        [sx - ISO_X, sy - h + ISO_Y],
      ].map((p) => p.join(",")).join(" ");
      // right face
      const right = [
        [sx + ISO_X, sy - h + ISO_Y],
        [sx, sy - h + ISO_Y * 2],
        [sx, sy + ISO_Y * 2],
        [sx + ISO_X, sy + ISO_Y],
      ].map((p) => p.join(",")).join(" ");
      // left face
      const left = [
        [sx - ISO_X, sy - h + ISO_Y],
        [sx, sy - h + ISO_Y * 2],
        [sx, sy + ISO_Y * 2],
        [sx - ISO_X, sy + ISO_Y],
      ].map((p) => p.join(",")).join(" ");

      columns.push(
        <g key={`${r}-${c}`}>
          <polygon points={left} fill={dark} />
          <polygon points={right} fill={mid} />
          <polygon points={top} fill={color} />
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

// Foot-shaped heatmap with contour coloring (right panel)
// plantar view, toes at top (row 0)
const FOOT_PATH =
  "M 60 10 C 70 8, 80 12, 82 22 C 84 30, 78 35, 72 32 " +
  "C 68 30, 64 28, 60 26 " +
  // pinky toe
  "C 54 24, 50 20, 52 14 C 53 10, 57 9, 60 10 Z " +
  "M 82 22 C 88 18, 96 20, 98 30 C 100 38, 94 44, 88 40 " +
  "C 84 37, 80 33, 78 30 C 79 27, 81 25, 82 22 Z " +
  "M 98 30 C 104 26, 112 28, 114 38 C 116 48, 110 54, 104 50 " +
  "C 100 47, 96 43, 94 40 C 96 37, 97 34, 98 30 Z " +
  "M 114 38 C 120 34, 128 36, 130 46 C 132 56, 126 62, 120 58 " +
  "C 116 55, 112 51, 110 48 C 112 45, 113 42, 114 38 Z " +
  "M 130 46 C 136 40, 144 42, 148 52 C 152 62, 146 70, 138 66 " +
  "C 132 63, 128 58, 126 54 C 128 51, 129 49, 130 46 Z";

// Simple plantar foot outline (blob shape)
const FOOT_OUTLINE =
  "M 90 8 C 110 6, 148 10, 158 28 C 168 46, 162 70, 148 86 " +
  "C 140 96, 130 104, 118 120 C 106 136, 98 155, 94 172 " +
  "C 90 188, 88 205, 86 218 C 84 230, 82 245, 88 258 " +
  "C 94 268, 110 275, 128 272 C 145 269, 158 258, 162 245 " +
  "C 166 232, 160 216, 154 202 C 148 188, 140 176, 134 162 " +
  "C 128 148, 124 134, 124 118 C 124 102, 130 88, 136 74 " +
  "C 142 60, 148 48, 144 34 C 140 20, 126 10, 110 8 C 102 7, 95 8, 90 8 Z";

// Toe bumps
const TOES = [
  "M 90 8 C 84 2, 76 2, 74 10 C 72 18, 78 24, 86 20 C 90 18, 92 14, 90 8 Z",
  "M 110 7 C 104 1, 96 1, 94 9 C 92 17, 98 23, 106 19 C 110 17, 112 13, 110 7 Z",
  "M 130 10 C 124 4, 116 4, 114 12 C 112 20, 118 26, 126 22 C 130 20, 132 16, 130 10 Z",
  "M 148 16 C 143 10, 135 10, 133 18 C 131 26, 136 32, 144 28 C 148 26, 150 22, 148 16 Z",
  "M 158 28 C 154 22, 147 23, 146 31 C 145 38, 150 44, 157 40 C 161 37, 162 33, 158 28 Z",
];

function FootHeatmap({ grid }: { grid: Grid }) {
  const W = 200;
  const H = 290;
  // map each cell in 16×8 grid to foot SVG space
  // row 0=toes (top, y~10), row 15=heel (bottom, y~260)
  // col 0=medial, col 7=lateral (x 60..180)
  const cells: ReactElement[] = [];
  const cellW = 14;
  const cellH = 16;

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const v = grid[r][c];
      const x = 70 + c * (cellW - 1.5);
      const y = 18 + r * (cellH - 0.5);
      cells.push(
        <rect
          key={`${r}-${c}`}
          x={x - cellW / 2}
          y={y - cellH / 2}
          width={cellW}
          height={cellH}
          fill={contourColor(v)}
          opacity={0.85 + v * 0.15}
        />
      );
    }
  }

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
      <defs>
        <clipPath id="foot-clip">
          <path d={FOOT_OUTLINE} />
          {TOES.map((d, i) => <path key={i} d={d} />)}
        </clipPath>
        {/* glow filter */}
        <filter id="foot-glow" x="-10%" y="-10%" width="120%" height="120%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      {/* cells clipped to foot */}
      <g clipPath="url(#foot-clip)">{cells}</g>
      {/* foot outline */}
      <path d={FOOT_OUTLINE} fill="none" stroke="rgba(148,163,184,0.5)" strokeWidth={1.5} />
      {TOES.map((d, i) => (
        <path key={i} d={d} fill="none" stroke="rgba(148,163,184,0.4)" strokeWidth={1} />
      ))}
    </svg>
  );
}

const LABELS: Record<MockProfile, string> = {
  normal: "균형 분포",
  parkinsons: "전족부 과부하 (발 끌림)",
  stroke: "편측 비대칭 하중",
  fall_risk: "외측 불안정",
};

const CONTOUR_LEGEND = [
  { color: "#1e3a5f", label: "0–20%" },
  { color: "#1d4ed8", label: "20–40%" },
  { color: "#10b981", label: "40–60%" },
  { color: "#f59e0b", label: "60–80%" },
  { color: "#ef4444", label: "80–100%" },
];

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

      <div className="flex gap-4 items-start">
        {/* 3D Isometric */}
        <div className="flex flex-col items-center gap-1">
          <span className="text-textMuted text-[10px]">3D 등각투영</span>
          <div className="overflow-hidden" style={{ width: 160, height: 160 }}>
            <IsoChart grid={grid} />
          </div>
        </div>

        {/* Foot heatmap */}
        <div className="flex flex-col items-center gap-1">
          <span className="text-textMuted text-[10px]">족저압 등고선</span>
          <FootHeatmap grid={grid} />
        </div>

        {/* Legend */}
        <div className="flex flex-col gap-1 mt-5 ml-1">
          {[...CONTOUR_LEGEND].reverse().map((l) => (
            <div key={l.label} className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm shrink-0" style={{ background: l.color }} />
              <span className="text-textMuted text-[9px]">{l.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
