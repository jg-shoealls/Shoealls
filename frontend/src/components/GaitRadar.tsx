"use client";

/**
 * GaitRadar — 순수 SVG 레이더 차트 (보행 특성 10축)
 */

interface AxisDef {
  key: string;
  label: string;
  min: number;
  max: number;
  invert: boolean;
}

const AXES: AxisDef[] = [
  { key: "gait_speed",          label: "속도",    min: 0,  max: 1.5,  invert: false },
  { key: "stride_regularity",   label: "규칙성",  min: 0,  max: 1,    invert: false },
  { key: "step_symmetry",       label: "대칭성",  min: 0,  max: 1,    invert: false },
  { key: "cadence",             label: "케이던스", min: 60, max: 130,  invert: false },
  { key: "arch_index",          label: "아치",    min: 0,  max: 0.45, invert: false },
  { key: "heel_pressure_ratio", label: "뒤꿈치",  min: 0,  max: 1,    invert: false },
  { key: "cop_sway",            label: "CoP",     min: 0,  max: 0.08, invert: true  },
  { key: "trunk_sway",          label: "몸통",    min: 0,  max: 12,   invert: true  },
  { key: "pressure_asymmetry",  label: "비대칭",  min: 0,  max: 0.4,  invert: true  },
  { key: "acceleration_rms",    label: "가속도",  min: 0,  max: 1.5,  invert: true  },
];

const CX = 150;
const CY = 150;
const R  = 110;
const N  = AXES.length;
const RINGS = 5;
const LABEL_RADIUS_FACTOR = 1.25;
const NORMAL_RADIUS_FACTOR = 0.75;

function toXY(angleRad: number, radius: number): [number, number] {
  return [
    CX + radius * Math.cos(angleRad),
    CY + radius * Math.sin(angleRad),
  ];
}

function axisAngle(i: number): number {
  // Start from top (−π/2) going clockwise
  return -Math.PI / 2 + (2 * Math.PI * i) / N;
}

function normalize(value: number, min: number, max: number, invert: boolean): number {
  const raw = (value - min) / (max - min);
  const clamped = Math.max(0, Math.min(1, raw));
  return invert ? 1 - clamped : clamped;
}

function polygonPath(radii: number[]): string {
  return radii
    .map((r, i) => {
      const [x, y] = toXY(axisAngle(i), r);
      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ") + " Z";
}

interface Props {
  features: Record<string, number>;
  label?: string;
}

export default function GaitRadar({ features, label }: Props) {
  const userRadii = AXES.map((axis) => {
    const val = features[axis.key] ?? axis.min;
    return normalize(val, axis.min, axis.max, axis.invert) * R;
  });

  const normalRadii = AXES.map(() => NORMAL_RADIUS_FACTOR * R);

  return (
    <div className="flex flex-col gap-2 items-center">
      {label && (
        <span className="text-textSec text-[12px] font-medium">{label}</span>
      )}

      <svg
        width="300"
        height="300"
        viewBox="0 0 300 300"
        aria-label="보행 레이더 차트"
      >
        {/* Concentric grid rings */}
        {Array.from({ length: RINGS }, (_, ri) => {
          const frac = (ri + 1) / RINGS;
          const ringRadii = AXES.map(() => frac * R);
          return (
            <path
              key={ri}
              d={polygonPath(ringRadii)}
              fill="none"
              stroke="#94A3B8"
              strokeOpacity={0.3}
              strokeWidth={1}
            />
          );
        })}

        {/* Axis lines */}
        {AXES.map((_, i) => {
          const [x, y] = toXY(axisAngle(i), R);
          return (
            <line
              key={i}
              x1={CX}
              y1={CY}
              x2={x.toFixed(2)}
              y2={y.toFixed(2)}
              stroke="#94A3B8"
              strokeOpacity={0.3}
              strokeWidth={1}
            />
          );
        })}

        {/* Normal reference polygon */}
        <path
          d={polygonPath(normalRadii)}
          fill="none"
          stroke="#3B82F6"
          strokeOpacity={0.4}
          strokeWidth={1.5}
          strokeDasharray="5 3"
        />

        {/* User data polygon */}
        <path
          d={polygonPath(userRadii)}
          fill="#3B82F6"
          fillOpacity={0.15}
          stroke="#3B82F6"
          strokeWidth={2}
          strokeLinejoin="round"
        />

        {/* User data dots */}
        {userRadii.map((r, i) => {
          const [x, y] = toXY(axisAngle(i), r);
          return (
            <circle
              key={i}
              cx={x.toFixed(2)}
              cy={y.toFixed(2)}
              r={3}
              fill="#3B82F6"
            />
          );
        })}

        {/* Axis labels */}
        {AXES.map((axis, i) => {
          const angle = axisAngle(i);
          const [x, y] = toXY(angle, R * LABEL_RADIUS_FACTOR);

          // Horizontal alignment
          let textAnchor: "start" | "middle" | "end" = "middle";
          const cosA = Math.cos(angle);
          if (cosA > 0.15) textAnchor = "start";
          else if (cosA < -0.15) textAnchor = "end";

          // Vertical offset
          const sinA = Math.sin(angle);
          const dy = sinA > 0.15 ? "0.9em" : sinA < -0.15 ? "0" : "0.35em";

          return (
            <text
              key={axis.key}
              x={x.toFixed(2)}
              y={y.toFixed(2)}
              dy={dy}
              textAnchor={textAnchor}
              fontSize={11}
              fill="#607085"
              fontFamily="inherit"
            >
              {axis.label}
            </text>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="flex items-center gap-4 text-[11px] text-textMuted">
        <span className="flex items-center gap-1.5">
          <svg width="18" height="6">
            <line x1="0" y1="3" x2="18" y2="3" stroke="#3B82F6" strokeWidth="1.5" strokeDasharray="5 3" />
          </svg>
          정상 기준
        </span>
        <span className="flex items-center gap-1.5">
          <svg width="18" height="6">
            <line x1="0" y1="3" x2="18" y2="3" stroke="#3B82F6" strokeWidth="2" />
          </svg>
          측정값
        </span>
      </div>
    </div>
  );
}
