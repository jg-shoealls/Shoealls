"use client";

import { useState } from "react";
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
  { icon: "Q", label: "QR 보고서",  href: "/report" },
];

// Mobile bottom nav shows 5 key items
const MOBILE_NAV = [
  { icon: "D", label: "대시보드", href: "/" },
  { icon: "G", label: "보행",     href: "/analysis" },
  { icon: "P", label: "파킨슨",   href: "/parkinson" },
  { icon: "H", label: "이력",     href: "/history" },
  { icon: "Q", label: "보고서",   href: "/report" },
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

function MenuIcon() {
  return (
    <svg width={18} height={18} viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round">
      <line x1={2} y1={5} x2={16} y2={5} />
      <line x1={2} y1={9} x2={16} y2={9} />
      <line x1={2} y1={13} x2={16} y2={13} />
    </svg>
  );
}

export default function Sidebar() {
  const pathname = usePathname();
  const { theme, toggle } = useTheme();
  const [drawerOpen, setDrawerOpen] = useState(false);

  const isActive = (href: string) =>
    pathname === href || (href !== "/" && pathname.startsWith(href));

  const navLink = (item: typeof NAV[0], onClick?: () => void) => {
    const active = isActive(item.href);
    return (
      <Link
        key={item.href}
        href={item.href}
        onClick={onClick}
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
  };

  return (
    <>
      {/* ── Desktop sidebar ────────────────────────────────── */}
      <aside className="hidden md:flex w-60 shrink-0 h-screen bg-surface flex-col sticky top-0 border-r border-border/50">
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-border">
          <div className="w-9 h-9 bg-blue rounded-lg flex items-center justify-center text-white font-bold text-lg">S</div>
          <div>
            <div className="font-semibold text-textPri text-[15px]">Shoealls</div>
            <div className="text-textSec text-[11px]">Smart Gait Care</div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-3 space-y-0.5 overflow-y-auto">
          {NAV.map((item) => navLink(item))}
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

      {/* ── Mobile: top bar + hamburger ─────────────────── */}
      <div className="md:hidden fixed top-0 left-0 right-0 z-40 h-12 bg-surface/95 border-b border-border/80 backdrop-blur-sm flex items-center px-4 gap-3">
        <button
          onClick={() => setDrawerOpen(true)}
          className="w-8 h-8 rounded-lg bg-card border border-border flex items-center justify-center text-textSec"
          aria-label="메뉴 열기"
        >
          <MenuIcon />
        </button>
        <div className="w-6 h-6 bg-blue rounded-md flex items-center justify-center text-white font-bold text-[11px]">S</div>
        <span className="text-textPri font-semibold text-[14px]">Shoealls</span>
      </div>

      {/* ── Mobile drawer overlay ───────────────────────── */}
      {drawerOpen && (
        <div
          className="md:hidden fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
          onClick={() => setDrawerOpen(false)}
        />
      )}

      {/* ── Mobile drawer ───────────────────────────────── */}
      <aside
        className={`md:hidden fixed top-0 left-0 h-full w-72 z-50 bg-surface flex flex-col border-r border-border/80 shadow-2xl
          transition-transform duration-300 ease-out
          ${drawerOpen ? "translate-x-0" : "-translate-x-full"}`}
      >
        {/* Drawer header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue rounded-lg flex items-center justify-center text-white font-bold text-[13px]">S</div>
            <div>
              <div className="font-semibold text-textPri text-[14px]">Shoealls</div>
              <div className="text-textSec text-[11px]">Smart Gait Care</div>
            </div>
          </div>
          <button
            onClick={() => setDrawerOpen(false)}
            className="w-7 h-7 rounded-lg bg-card border border-border flex items-center justify-center text-textMuted"
            aria-label="닫기"
          >
            <svg width={12} height={12} viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round">
              <line x1={1} y1={1} x2={11} y2={11} />
              <line x1={11} y1={1} x2={1} y2={11} />
            </svg>
          </button>
        </div>

        {/* Drawer nav */}
        <nav className="flex-1 p-3 space-y-0.5 overflow-y-auto">
          {NAV.map((item) => navLink(item, () => setDrawerOpen(false)))}
        </nav>

        {/* Theme toggle in drawer */}
        <div className="px-3 pb-3">
          <button
            onClick={toggle}
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg border border-border text-textSec hover:bg-card transition-colors text-[12px]"
          >
            <span className="flex items-center gap-2">
              {theme === "dark" ? <MoonIcon /> : <SunIcon />}
              <span>{theme === "dark" ? "다크 모드" : "라이트 모드"}</span>
            </span>
            <div className={`relative w-9 h-5 rounded-full transition-colors ${theme === "light" ? "bg-blue" : "bg-border"}`}>
              <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                theme === "light" ? "translate-x-4" : "translate-x-0.5"
              }`} />
            </div>
          </button>
        </div>
      </aside>

      {/* ── Mobile bottom tab bar ───────────────────────── */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-surface/95 border-t border-border/80 backdrop-blur-sm flex items-center justify-around px-2 py-1.5 safe-area-inset-bottom">
        {MOBILE_NAV.map((item) => {
          const active = isActive(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex flex-col items-center gap-0.5 px-3 py-1.5 rounded-xl transition-colors min-w-[52px] ${
                active ? "text-blue" : "text-textMuted"
              }`}
            >
              <span className={`w-6 h-6 rounded-lg text-[12px] flex items-center justify-center font-bold transition-colors ${
                active ? "bg-blue/15 text-blue" : "text-textMuted"
              }`}>
                {item.icon}
              </span>
              <span className={`text-[9px] font-medium ${active ? "text-blue" : "text-textMuted"}`}>
                {item.label}
              </span>
            </Link>
          );
        })}
      </nav>
    </>
  );
}
