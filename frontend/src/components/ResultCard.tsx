import type { ReactNode } from "react";

export function ResultCard({
  title,
  badge,
  accentColor,
  children,
  isDemo = false,
}: {
  title: string;
  badge: string;
  accentColor: string;
  children: ReactNode;
  isDemo?: boolean;
}) {
  return (
    <section className="bg-card rounded-xl overflow-hidden flex flex-col border border-border/50">
      <div className="h-1" style={{ background: accentColor }} />
      <div className="p-5 flex-1">
        <div className="flex items-center gap-2 mb-3">
          <Badge label={badge} color={accentColor} />
          {isDemo && <Badge label="데모" color="#F59E0B" />}
        </div>
        <h3 className="text-textPri font-semibold text-[17px] mb-4">{title}</h3>
        {children}
      </div>
    </section>
  );
}

export function Badge({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="px-2.5 py-0.5 rounded-full text-[11px] font-medium"
      style={{ color, background: `${color}28` }}
    >
      {label}
    </span>
  );
}

export function ProgressBar({
  pct,
  color,
  label,
  valueLabel,
}: {
  pct: number;
  color: string;
  label: string;
  valueLabel: string;
}) {
  const normalized = Math.max(0, Math.min(100, pct * 100));

  return (
    <div>
      {(label || valueLabel) && (
        <div className="flex justify-between mb-1.5 gap-3">
          <span className="text-textSec text-[12px] truncate">{label}</span>
          <span className="text-textPri text-[12px] font-semibold shrink-0">{valueLabel}</span>
        </div>
      )}
      <div className="h-1.5 bg-surface rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${normalized}%`, background: color }}
        />
      </div>
    </div>
  );
}
