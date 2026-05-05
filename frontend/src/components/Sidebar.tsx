"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";

const NAV = [
  { icon: "⊞", label: "대시보드",    href: "/dashboard" },
  { icon: "◎", label: "보행 분석",   href: "/analysis" },
  { icon: "⚕", label: "질환 위험도", href: "/disease" },
  { icon: "⚑", label: "부상 예측",   href: "/injury" },
  { icon: "⬡", label: "AI 추론",     href: "/reasoning" },
  { icon: "≡", label: "분석 이력",   href: "/history" },
];

export default function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="w-64 shrink-0 h-screen glass border-r-0 border-white/5 flex flex-col sticky top-0 relative z-20">
      <div className="flex items-center justify-center px-6 py-6 border-b border-border">
        <img src="/logo.png" alt="Shoealls Logo" className="h-12 w-auto" />
      </div>

      <nav className="flex-1 px-4 py-6 space-y-1.5">
        {NAV.map((item) => {
          const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.label}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-4 rounded-xl text-lg transition-all duration-300 ${
                active
                  ? "bg-primaryBlue/10 text-primaryBlue font-semibold shadow-sm border border-primaryBlue/20"
                  : "text-textSec hover:bg-black/5 dark:hover:bg-white/5 hover:text-textPri border border-transparent"
              }`}
            >
              <span className={`w-8 text-center text-lg ${active ? 'text-primaryBlue' : 'text-textMuted'}`}>
                {item.icon}
              </span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="m-5 p-4 rounded-xl glass-card border border-medicalTeal/20 bg-medicalTeal/5 relative overflow-hidden group">
        <div className="relative z-10 flex items-center gap-2.5 mb-1.5">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-medicalTeal opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-medicalTeal"></span>
          </span>
          <span className="text-medicalTeal text-base font-semibold tracking-wide">API Server Active</span>
        </div>
        <div className="relative z-10 text-textMuted text-sm ml-5">localhost:8000 · v0.1.0</div>
      </div>

      <div className="px-5 pb-5 mt-auto flex justify-end">
        <ThemeToggle />
      </div>
    </aside>
  );
}
