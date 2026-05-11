"use client";

import { useEffect, useState } from "react";
import type { MockProfile } from "@/lib/mockSensorData";

const ROWS = 16;
const COLS = 8;

type Grid = number[][];

// Anatomical pressure baseline per profile (row 0=toes, row 15=heel)
const BASE: Record<MockProfile, (r: number, c: number) => number> = {
  normal: (r, c) => {
    const heel = r > 11 ? 0.85 : 0;
    const forefoot = r < 4 ? 0.75 : 0;
    const arch = r >= 4 && r <= 11 ? 0.15 : 0;
    const lateral = c >= 2 && c <= 5 ? 0.1 : 0;
    return heel + forefoot + arch + lateral;
  },
  parkinsons: (r, c) => {
    // shuffling: high forefoot, low heel lift
    const forefoot = r < 6 ? 0.9 : 0;
    const heel = r > 13 ? 0.3 : 0;
    const medial = c >= 3 && c <= 5 ? 0.15 : 0;
    return forefoot + heel + medial;
  },
  stroke: (r, c) => {
    // asymmetric: right side (col 4-7) dominates
    const side = c >= 4 ? 0.9 : 0.3;
    const heel = r > 10 ? 0.5 : 0;
    const toe = r < 3 ? 0.4 : 0;
    return (side + heel + toe) * 0.6;
  },
  fall_risk: (r, c) => {
    // lateral instability
    const lateral = (c < 2 || c > 5) ? 0.75 : 0.2;
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

function valToColor(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  if (t < 0.25) {
    const s = t / 0.25;
    return `rgb(${Math.round(15 + 44 * s)},${Math.round(23 + 107 * s)},${Math.round(42 + 204 * s)})`;
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    return `rgb(${Math.round(59 - 43 * s)},${Math.round(130 + 55 * s)},${Math.round(246 - 202 * s)})`;
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    return `rgb(${Math.round(16 + 229 * s)},${Math.round(185 - 27 * s)},${Math.round(44 - 33 * s)})`;
  } else {
    const s = (t - 0.75) / 0.25;
    return `rgb(${Math.round(245 - 6 * s)},${Math.round(158 - 90 * s)},${Math.round(11 - 11 * s)})`;
  }
}

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

  const LABELS: Record<MockProfile, string> = {
    normal: "균형 분포",
    parkinsons: "전족부 과부하 (발 끌림)",
    stroke: "편측 비대칭 하중",
    fall_risk: "외측 불안정",
  };

  return (
    <div className="bg-card rounded-xl border border-border p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-textPri font-semibold text-[13px]">족저압 분포 히트맵</span>
        <span className="text-[11px] text-amber font-medium">{LABELS[profile]}</span>
      </div>
      <div className="flex gap-3 justify-center">
        {["왼발", "오른발"].map((side) => (
          <div key={side}>
            <div className="text-textMuted text-[10px] text-center mb-1">{side}</div>
            <div
              className="grid gap-px rounded-lg overflow-hidden"
              style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)`, width: 96, height: 128 }}
            >
              {grid.map((row, r) =>
                row.map((val, c) => (
                  <div
                    key={`${r}-${c}`}
                    style={{
                      background: valToColor(side === "오른발" ? val * 0.85 + (Math.random() - 0.5) * 0.05 : val),
                      transition: "background 0.15s ease",
                    }}
                  />
                ))
              )}
            </div>
          </div>
        ))}
        <div className="flex flex-col justify-between ml-2">
          <span className="text-[9px] text-red-400">High</span>
          <div
            className="w-2 flex-1 rounded my-1"
            style={{ background: "linear-gradient(to bottom, #EF4444, #F59E0B, #10B981, #3B82F6, #0F172A)" }}
          />
          <span className="text-[9px] text-blue-400">Low</span>
        </div>
      </div>
    </div>
  );
}
