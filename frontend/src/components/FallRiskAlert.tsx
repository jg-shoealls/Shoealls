import type { FallRiskResult } from "@/types/sensor";

const LEVEL_STYLES: Record<FallRiskResult["level"], { color: string; bg: string; border: string }> = {
  low: { color: "#10B981", bg: "rgba(16,185,129,0.10)", border: "rgba(16,185,129,0.28)" },
  watch: { color: "#3B82F6", bg: "rgba(59,130,246,0.10)", border: "rgba(59,130,246,0.28)" },
  elevated: { color: "#F59E0B", bg: "rgba(245,158,11,0.12)", border: "rgba(245,158,11,0.32)" },
  high: { color: "#EF4444", bg: "rgba(239,68,68,0.12)", border: "rgba(239,68,68,0.34)" },
};

export function FallRiskAlert({ risk }: { risk: FallRiskResult }) {
  const style = LEVEL_STYLES[risk.level];

  return (
    <section className="rounded-xl p-5 border" style={{ background: style.bg, borderColor: style.border }}>
      <div className="flex items-start justify-between gap-5">
        <div>
          <div className="text-textMuted text-[12px] mb-1">낙상 위험 알림</div>
          <div className="flex items-center gap-3">
            <span className="text-2xl font-bold" style={{ color: style.color }}>
              {risk.label}
            </span>
            <span className="text-textPri text-[18px] font-semibold">{Math.round(risk.score * 100)}%</span>
          </div>
        </div>
        <div className="text-right max-w-xl">
          <p className="text-textSec text-[13px] leading-relaxed">{risk.recommendation}</p>
          <p className="text-textMuted text-[11px] mt-1">
            본 결과는 진단이 아니라 센서 기반 위험 징후 모니터링입니다.
          </p>
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {risk.reasons.map((reason) => (
          <span key={reason} className="px-2.5 py-1 rounded-md text-[12px]" style={{ color: style.color, background: `${style.color}20` }}>
            {reason}
          </span>
        ))}
      </div>
    </section>
  );
}
