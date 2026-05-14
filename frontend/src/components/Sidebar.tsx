"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "@/lib/useTheme";

const NAV = [
  { icon: "D", label: "대시보드",    href: "/" },
  { icon: "G", label: "보행 분석",   href: "/analysis" },
  { icon: "R", label: "위험 징후",   href: "/disease" },
  { icon: "F", label: "낙상/부상",   href: "/injury" },
  { icon: "A", label: "AI 추론",     href: "/reasoning" },
  { icon: "N", label: "노드 그래프", href: "/graph" },
  { icon: "P", label: "파킨슨 진단", href: "/parkinson" },
  { icon: "H", label: "분석 이력",   href: "/history" },
];

function SunIcon() {
  return (
    <svg width={15} height={15} viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth={1.4} strokeLinecap="round">
      <circle cx={7.5} cy={7.5} r={3} />
      <line x1={7.5} y1={0.5} x2={7.5} y2={2.5} />
      <line x1={7.5} y1={12.5} x2={7.5} y2={14.5} />
      <line x1={0.5} y1={7.5} x2={2.5} y2={7.5} />
      <line x1={12.5} y1={7.5} x2={14.5} y2={7.5} />
      <line x1={2.5} y1={2.5} x2={4} y2={4} />
      <line x1={11} y1={11} x2={12.5} y2={12.5} />
      <line x1={2.5} y1={12.5} x2={4} y2={11} />
      <line x1={11} y1={4} x2={12.5} y2={2.5} />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width={15} height={15} viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth={1.4} strokeLinecap="round">
      <path d="M 11.5 10 Q 7 12 4 8 Q 2 4 5 1.5 Q 3 5 5 8 Q 7 11 11.5 10 Z" />
    </svg>
  );
}

export default function Sidebar() {
  const pathname = usePathname();
  const { theme, toggle } = useTheme();

  return (
    <aside className="w-60 shrink-0 h-screen bg-surface flex flex-col sticky top-0 border-r border-border/50">
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-border">
        <div className="w-9 h-9 bg-blue rounded-lg flex items-center justify-center text-white font-bold text-lg">
          S
        </div>
        <div>
          <div className="font-semibold text-textPri text-[15px]">Shoealls</div>
          <div className="text-textSec text-[11px]">Smart Gait Care</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-3 space-y-0.5 overflow-y-auto">
        {NAV.map((item) => {
          const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-[13px] transition-colors ${
                active ? "bg-blue/15 text-blue font-semibold" : "text-textSec hover:bg-card hover:text-textPri"
              }`}
            >
              <span className="w-5 h-5 rounded-md bg-card border border-border text-[10px] flex items-center justify-center shrink-0">
                {item.icon}
              </span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Theme toggle */}
      <div className="px-3 pb-2">
        <button
          onClick={toggle}
          className="w-full flex items-center justify-between px-3 py-2 rounded-lg border border-border text-textSec hover:bg-card hover:text-textPri transition-colors text-[12px]"
          aria-label="테마 변경"
        >
          <span className="flex items-center gap-2">
            {theme === "dark" ? <MoonIcon /> : <SunIcon />}
            <span>{theme === "dark" ? "다크 모드" : "라이트 모드"}</span>
          </span>
          {/* Toggle track */}
          <div className={`relative w-9 h-5 rounded-full transition-colors ${theme === "light" ? "bg-blue" : "bg-border"}`}>
            <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
              theme === "light" ? "translate-x-4" : "translate-x-0.5"
            }`} />
          </div>
        </button>
      </div>

      {/* API status */}
      <div className="m-3 mt-0 p-3 rounded-xl bg-green/10 border border-green/20">
        <div className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full bg-green inline-block" />
          <span className="text-green text-[12px] font-semibold">API 연결 준비</span>
        </div>
        <div className="text-textMuted text-[10px]">{process.env.NEXT_PUBLIC_API_URL || "localhost:8000"} · v0.1.0</div>
      </div>
    </aside>
  );
}
