import { ProgressBar, ResultCard } from "@/components/ResultCard";
import type { GaitReport } from "@/types/sensor";

const COLORS = {
  blue: "#3B82F6",
  green: "#10B981",
  amber: "#F59E0B",
  purple: "#AF65FA",
};

export function GaitReportCard({ report }: { report: GaitReport }) {
  const metrics = [
    { label: "좌우 대칭성", value: report.symmetryPct, color: COLORS.blue },
    { label: "보행 리듬", value: report.rhythmPct, color: COLORS.green },
    { label: "동적 안정성", value: report.stabilityPct, color: COLORS.amber },
    { label: "체중 이동", value: report.weightTransferPct, color: COLORS.purple },
  ];

  return (
    <ResultCard title="보행 리포트" badge="Rule-based report" accentColor={COLORS.blue}>
      <div className="grid grid-cols-[120px_1fr] gap-5">
        <div className="w-28 h-28 rounded-full bg-surface flex flex-col items-center justify-center border border-blue/40">
          <span className="text-textMuted text-[11px]">보행 점수</span>
          <span className="text-textPri text-3xl font-bold">{report.score}</span>
          <span className="text-textMuted text-[11px]">/ 100</span>
        </div>
        <div className="space-y-3">
          {metrics.map((metric) => (
            <ProgressBar
              key={metric.label}
              pct={metric.value / 100}
              color={metric.color}
              label={metric.label}
              valueLabel={`${metric.value}%`}
            />
          ))}
        </div>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3">
        <div className="bg-surface rounded-lg p-3">
          <div className="text-textMuted text-[11px] mb-1">검출 이벤트</div>
          <div className="text-textPri font-semibold text-[15px]">{report.events.length}건</div>
        </div>
        <div className="bg-surface rounded-lg p-3">
          <div className="text-textMuted text-[11px] mb-1">이상 징후</div>
          <div className="text-textPri font-semibold text-[15px]">{report.abnormalPatterns.length}건</div>
        </div>
      </div>

      {report.abnormalPatterns.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {report.abnormalPatterns.map((pattern) => (
            <span key={pattern} className="px-2.5 py-1 rounded-md bg-amber/15 text-amber text-[12px]">
              {pattern}
            </span>
          ))}
        </div>
      )}
    </ResultCard>
  );
}
