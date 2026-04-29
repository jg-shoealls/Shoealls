/** 결과 카드 컨테이너 */
export function ResultCard({
  title,
  badge,
  accentColor,
  children,
  isDemo = false,
  delay = "0s",
}: {
  title: string;
  badge: string;
  accentColor: string;
  children: React.ReactNode;
  isDemo?: boolean;
  delay?: string;
}) {
  return (
    <div 
      className="glass-card rounded-[20px] overflow-hidden flex flex-col relative animate-fade-in-up opacity-0"
      style={{ animationDelay: delay }}
    >
      {/* 상단 컬러 바 */}
      <div 
        className="h-[4px] w-full" 
        style={{ background: accentColor }} 
      />
      
      <div className="p-6 flex-1 relative z-10">
        <div className="flex items-center gap-2 mb-4">
          <Badge label={badge} color={accentColor} />
          {isDemo && <Badge label="Demo" color="#F59E0B" />}
        </div>
        <h3 className="text-textPri font-bold text-[18px] mb-5 tracking-wide">{title}</h3>
        {children}
      </div>
    </div>
  );
}

/** 소형 배지 */
export function Badge({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="px-3 py-1 rounded-full text-[11px] font-bold tracking-wider uppercase border"
      style={{ 
        color, 
        background: `${color}10`,
        borderColor: `${color}30`
      }}
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
    <div className="group">
      <div className="flex justify-between mb-2">
        <span className="text-textSec text-[13px] group-hover:text-textPri transition-colors">{label}</span>
        <span className="text-textPri text-[13px] font-bold tracking-wide">{valueLabel}</span>
      </div>
      <div className="h-2 bg-black/10 dark:bg-black/40 rounded-full overflow-hidden border border-border/50 relative">
        <div
          className="h-full rounded-full transition-all duration-1000 ease-out relative"
          style={{ 
            width: `${Math.min(pct * 100, 100)}%`, 
            background: color
          }}
        >
        </div>
      </div>
    </div>
  );
}
