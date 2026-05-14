"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { icon: "D", label: "대시보드", href: "/" },
  { icon: "G", label: "보행 분석", href: "/analysis" },
  { icon: "R", label: "위험 징후", href: "/disease" },
  { icon: "F", label: "낙상/부상", href: "/injury" },
  { icon: "A", label: "AI 추론", href: "/reasoning" },
  { icon: "N", label: "노드 그래프", href: "/graph" },
  { icon: "P", label: "파킨슨 진단", href: "/parkinson" },
  { icon: "H", label: "분석 이력", href: "/history" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-60 shrink-0 h-screen bg-surface flex flex-col sticky top-0">
      <div className="flex items-center gap-3 px-5 py-5 border-b border-border">
        <div className="w-9 h-9 bg-blue rounded-lg flex items-center justify-center text-white font-bold text-lg">
          S
        </div>
        <div>
          <div className="font-semibold text-textPri text-[15px]">Shoealls</div>
          <div className="text-textSec text-[11px]">Smart Gait Care</div>
        </div>
      </div>

      <nav className="flex-1 p-3 space-y-1">
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
              <span className="w-5 h-5 rounded-md bg-card border border-border text-[10px] flex items-center justify-center">
                {item.icon}
              </span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="m-3 p-3 rounded-xl bg-green/10 border border-green/20">
        <div className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full bg-green inline-block" />
          <span className="text-green text-[12px] font-semibold">API 연결 준비</span>
        </div>
        <div className="text-textMuted text-[10px]">{process.env.NEXT_PUBLIC_API_URL || "localhost:8000"} · v0.1.0</div>
      </div>
    </aside>
  );
}
