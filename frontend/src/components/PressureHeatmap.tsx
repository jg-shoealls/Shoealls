"use client";

/**
 * PressureHeatmap — 16×8 족저압 그리드를 색상 히트맵으로 렌더링
 */

const STOPS: [number, number, number][] = [
  [30, 58, 138],   // dark blue  0.0
  [6, 182, 212],   // cyan       0.25
  [34, 197, 94],   // green      0.5
  [234, 179, 8],   // yellow     0.75
  [220, 38, 38],   // red        1.0
];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function valueToColor(v: number): string {
  const clamped = Math.max(0, Math.min(1, v));
  const scaled = clamped * (STOPS.length - 1);
  const lo = Math.floor(scaled);
  const hi = Math.min(lo + 1, STOPS.length - 1);
  const t = scaled - lo;

  const [r, g, b] = [
    Math.round(lerp(STOPS[lo][0], STOPS[hi][0], t)),
    Math.round(lerp(STOPS[lo][1], STOPS[hi][1], t)),
    Math.round(lerp(STOPS[lo][2], STOPS[hi][2], t)),
  ];

  return `rgb(${r},${g},${b})`;
}

const LEGEND_STOPS = 40;

interface Props {
  pressure: number[][];
  label?: string;
}

export default function PressureHeatmap({ pressure, label }: Props) {
  // pressure is [rows][cols] — expect up to 16 rows × 8 cols
  const rows = pressure.length;
  const cols = pressure[0]?.length ?? 0;

  return (
    <div className="flex flex-col gap-2">
      {label && (
        <span className="text-textSec text-[12px] font-medium">{label}</span>
      )}

      {/* Grid */}
      <div
        className="w-full"
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${cols}, 1fr)`,
          gap: "2px",
        }}
      >
        {Array.from({ length: rows }, (_, row) =>
          Array.from({ length: cols }, (_, col) => {
            const v = pressure[row]?.[col] ?? 0;
            const clamped = Math.max(0, Math.min(1, v));
            const opacity = 0.15 + clamped * 0.85;
            return (
              <div
                key={`${row}-${col}`}
                style={{
                  aspectRatio: "1",
                  backgroundColor: valueToColor(clamped),
                  opacity,
                  borderRadius: "2px",
                }}
              />
            );
          })
        )}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-2 mt-1">
        <span className="text-textMuted text-[10px] shrink-0">저</span>
        <div
          className="flex-1 h-2 rounded-full overflow-hidden"
          style={{
            background: `linear-gradient(to right, ${Array.from(
              { length: LEGEND_STOPS },
              (_, i) => valueToColor(i / (LEGEND_STOPS - 1))
            ).join(", ")})`,
          }}
        />
        <span className="text-textMuted text-[10px] shrink-0">고</span>
      </div>
    </div>
  );
}
