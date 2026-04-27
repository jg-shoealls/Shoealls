"use client";

/**
 * RiskGauge — 반원 아크 게이지 (위험도 시각화)
 */

interface Props {
  score: number;   // 0–1
  label: string;
  size?: number;   // default 180
}

function scoreColor(score: number): string {
  if (score < 0.3) return "#10B981"; // green
  if (score < 0.6) return "#F59E0B"; // amber
  return "#EF4444";                   // red
}

export default function RiskGauge({ score, label, size = 180 }: Props) {
  const clamped  = Math.max(0, Math.min(1, score));
  const width    = size;
  const height   = Math.round(size * 0.6);
  const cx       = size / 2;
  const cy       = Math.round(size * 0.55);
  const r        = Math.round(size * 0.38);
  const sw       = 14;

  const totalLen = Math.PI * r;
  const dashArray  = `${totalLen} ${totalLen}`;
  const dashOffset = totalLen * (1 - clamped);

  const arcPath = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;

  const color = scoreColor(clamped);
  const pct   = (clamped * 100).toFixed(0);

  // Endpoint label font size scales with gauge size
  const endLabelSize = Math.max(9, Math.round(size * 0.065));
  const scoreFontSize = Math.max(18, Math.round(size * 0.17));
  const labelFontSize = Math.max(10, Math.round(size * 0.075));

  return (
    <div className="flex flex-col items-center">
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        aria-label={`${label} 위험도 ${pct}%`}
      >
        {/* Background arc */}
        <path
          d={arcPath}
          fill="none"
          stroke="#32425B"
          strokeWidth={sw}
          strokeLinecap="round"
        />

        {/* Foreground arc */}
        <path
          d={arcPath}
          fill="none"
          stroke={color}
          strokeWidth={sw}
          strokeLinecap="round"
          strokeDasharray={dashArray}
          strokeDashoffset={dashOffset}
          style={{
            transition: "stroke-dashoffset 0.8s ease, stroke 0.4s",
          }}
        />

        {/* Score text */}
        <text
          x={cx}
          y={cy - r * 0.08}
          textAnchor="middle"
          dominantBaseline="auto"
          fontSize={scoreFontSize}
          fontWeight="bold"
          fill="#F1F5F9"
          fontFamily="inherit"
        >
          {pct}%
        </text>

        {/* Label */}
        <text
          x={cx}
          y={cy + r * 0.18}
          textAnchor="middle"
          dominantBaseline="hanging"
          fontSize={labelFontSize}
          fill="#94A3B8"
          fontFamily="inherit"
        >
          {label}
        </text>

        {/* 0 label at left end */}
        <text
          x={cx - r - sw / 2 - 2}
          y={cy + endLabelSize * 0.4}
          textAnchor="end"
          fontSize={endLabelSize}
          fill="#607085"
          fontFamily="inherit"
        >
          0
        </text>

        {/* 100 label at right end */}
        <text
          x={cx + r + sw / 2 + 2}
          y={cy + endLabelSize * 0.4}
          textAnchor="start"
          fontSize={endLabelSize}
          fill="#607085"
          fontFamily="inherit"
        >
          100
        </text>
      </svg>
    </div>
  );
}
