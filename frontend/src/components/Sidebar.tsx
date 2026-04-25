"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const NAV = [
  { icon: "⊞", label: "대시보드",    href: "/" },
  { icon: "◎", label: "보행 분석",   href: "/analysis" },
  { icon: "⚕", label: "질환 위험도", href: "/disease" },
  { icon: "⚑", label: "부상 예측",   href: "/injury" },
  { icon: "⬡", label: "AI 추론",     href: "/reasoning" },
  { icon: "≡", label: "분석 이력",   href: "/history" },
];

type ApiStatus = "checking" | "ok" | "error";

export default function Sidebar() {
  const pathname = usePathname();
  const [status, setStatus] = useState<ApiStatus>("checking");
  const [version, setVersion] = useState<string>("...");

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await api.health();
        if (!cancelled) {
          setStatus("ok");
          setVersion(res.version ?? "");
        }
      } catch {
        if (!cancelled) setStatus("error");
      }
    };
    check();
    const id = setInterval(check, 30_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const statusColor  = status === "ok" ? "green" : status === "error" ? "red" : "amber";
  const statusLabel  = status === "ok" ? "API 서버 정상" : status === "error" ? "API 서버 오프라인" : "연결 확인 중…";
  const dotBg        = `bg-${statusColor}`;
  const cardBg       = `bg-${statusColor}/10 border-${statusColor}/20`;
  const textColor    = `text-${statusColor}`;

  return (
    <aside className="w-60 shrink-0 h-screen bg-surface flex flex-col sticky top-0">
      <div className="flex items-center gap-3 px-5 py-5 border-b border-border">
        <div className="w-9 h-9 bg-blue rounded-[10px] flex items-center justify-center text-white font-bold text-lg">
          S
        </div>
        <div>
          <div className="font-semibold text-textPri text-[15px]">Shoealls</div>
          <div className="text-textSec text-[11px]">Gait AI</div>
        </div>
      </div>

      <nav className="flex-1 p-3 space-y-1">
        {NAV.map((item) => {
          const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.label}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-[13px] transition-colors ${
                active
                  ? "bg-blue/15 text-blue font-semibold"
                  : "text-textSec hover:bg-card hover:text-textPri"
              }`}
            >
              <span className="w-5 text-center">{item.icon}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className={`m-3 p-3 rounded-xl border ${cardBg}`}>
        <div className="flex items-center gap-2 mb-1">
          <span className={`w-2 h-2 rounded-full inline-block ${dotBg} ${status === "checking" ? "animate-pulse" : ""}`} />
          <span className={`text-[12px] font-semibold ${textColor}`}>{statusLabel}</span>
        </div>
        <div className="text-textMuted text-[10px]">
          {process.env.NEXT_PUBLIC_API_URL || "localhost:8000"}
          {version ? ` · v${version}` : ""}
        </div>
      </div>
    </aside>
  );
}
