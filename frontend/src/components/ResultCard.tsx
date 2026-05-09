/** 결과 카드 컨테이너 */
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
  children: React.ReactNode;
  isDemo?: boolean;
}) {
  return (
    <div className="bg-card rounded-2xl overflow-hidden flex flex-col">
      {/* 상단 컬러 바 */}
      <div className="h-1" style={{ background: accentColor }} />
      <div className="p-5 flex-1">
        <div className="flex items-center gap-2 mb-3">
          <Badge label={badge} color={accentColor} />
          {isDemo && <Badge label="데모" color="#F59E0B" />}
        </div>
        <h3 className="text-textPri font-semibold text-[17px] mb-4">{title}</h3>
        {children}
      </div>
    </div>
  );
}

/** 소형 배지 */
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

/** 진행 바 */
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
  return (
    <div>
      <div className="flex justify-between mb-1.5">
        <span className="text-textSec text-[12px]">{label}</span>
        <span className="text-textPri text-[12px] font-semibold">{valueLabel}</span>
      </div>
      <div className="h-1.5 bg-surface rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${Math.min(pct * 100, 100)}%`, background: color }}
        />
      </div>
    </div>
  );
}
