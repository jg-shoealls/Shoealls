"use client";

/**
 * CopTrajectory — 압력 중심(CoP) 궤적 시각화 (SVG)
 */

const SVG_W = 280;
const SVG_H = 200;
const PAD_X = 20;
const PAD_Y = 10;
const DRAW_W = 240;
const DRAW_H = 180;
const N = 60;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function synthesizePoints(features: Record<string, number>): [number, number][] {
  const points: [number, number][] = [];
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    const mlOffset = (features.ml_index ?? 0) * 0.15;
    const sway = features.cop_sway ?? 0.02;
    const x = 0.5 + mlOffset + sway * 30 * Math.sin(t * Math.PI * 5);
    const y = 0.08 + t * 0.84 + sway * 15 * Math.cos(t * Math.PI * 4);
    points.push([clamp(x, 0.05, 0.95), clamp(y, 0.05, 0.95)]);
  }
  return points;
}

function toSvg([x, y]: [number, number]): [number, number] {
  return [PAD_X + x * DRAW_W, PAD_Y + y * DRAW_H];
}

/** Lerp between two hex-style rgb triplets for gradient segments */
function segmentColor(t: number): string {
  // blue #3B82F6 → purple #AF65FA
  const r = Math.round(59  + (175 - 59)  * t);
  const g = Math.round(130 + (101 - 130) * t);
  const b = Math.round(246 + (250 - 246) * t);
  return `rgb(${r},${g},${b})`;
}

interface Props {
  features: Record<string, number>;
  label?: string;
}

export default function CopTrajectory({ features, label }: Props) {
  const points = synthesizePoints(features);
  const svgPts = points.map(toSvg);

  const copSway = (features.cop_sway ?? 0).toFixed(3);
  const mlIndex = (features.ml_index ?? 0).toFixed(3);

  const [startX, startY] = svgPts[0];
  const [endX, endY]     = svgPts[svgPts.length - 1];

  // Foot outline: centered rounded rect
  const footX = PAD_X + DRAW_W * 0.5 - 28;
  const footY = PAD_Y + DRAW_H * 0.05;
  const footW = 56;
  const footH = DRAW_H * 0.9;

  return (
    <div className="flex flex-col gap-2">
      {label && (
        <span className="text-textSec text-[12px] font-medium">{label}</span>
      )}

      <svg
        width={SVG_W}
        height={SVG_H}
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        aria-label="CoP 궤적"
      >
        {/* Foot outline */}
        <rect
          x={footX}
          y={footY}
          width={footW}
          height={footH}
          rx={20}
          ry={20}
          fill="#F1F5F9"
          fillOpacity={0.08}
          stroke="#94A3B8"
          strokeOpacity={0.15}
          strokeWidth={1}
        />

        {/* Path segments with gradient */}
        {svgPts.slice(0, -1).map(([x1, y1], i) => {
          const [x2, y2] = svgPts[i + 1];
          const t = i / (N - 2);
          const opacity = 0.25 + t * 0.75;
          return (
            <line
              key={i}
              x1={x1.toFixed(2)}
              y1={y1.toFixed(2)}
              x2={x2.toFixed(2)}
              y2={y2.toFixed(2)}
              stroke={segmentColor(t)}
              strokeWidth={2}
              strokeOpacity={opacity}
              strokeLinecap="round"
            />
          );
        })}

        {/* Start dot (blue) */}
        <circle
          cx={startX.toFixed(2)}
          cy={startY.toFixed(2)}
          r={5}
          fill="#3B82F6"
          stroke="#1E2940"
          strokeWidth={1.5}
        />

        {/* End dot (purple) */}
        <circle
          cx={endX.toFixed(2)}
          cy={endY.toFixed(2)}
          r={5}
          fill="#AF65FA"
          stroke="#1E2940"
          strokeWidth={1.5}
        />

        {/* Labels */}
        <text x={PAD_X + 4} y={SVG_H - 6} fontSize={9} fill="#607085">
          CoP Sway: {copSway}
        </text>
        <text
          x={SVG_W - PAD_X - 4}
          y={SVG_H - 6}
          fontSize={9}
          fill="#607085"
          textAnchor="end"
        >
          ML: {mlIndex}
        </text>
      </svg>

      {/* Legend */}
      <div className="flex items-center gap-4 text-[10px] text-textMuted">
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-blue inline-block" />
          시작
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-purple inline-block" />
          종료
        </span>
      </div>
    </div>
  );
}
