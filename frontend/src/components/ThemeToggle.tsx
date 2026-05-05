"use client";

import { useEffect, useState } from "react";

type Theme = "dark" | "light";

export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("dark");

  // On mount: read persisted preference
  useEffect(() => {
    const stored = (localStorage.getItem("theme") as Theme | null) ?? "dark";
    setTheme(stored);
    document.documentElement.setAttribute("data-theme", stored);
  }, []);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
  }

  const isDark  = theme === "dark";
  const icon    = isDark ? "☀" : "◑";
  const tooltip = isDark ? "라이트 모드" : "다크 모드";

  return (
    <button
      onClick={toggle}
      title={tooltip}
      aria-label={tooltip}
      className="w-8 h-8 rounded-lg bg-card border border-border text-textSec hover:text-textPri transition-colors flex items-center justify-center"
    >
      <span className="text-[15px] leading-none select-none">{icon}</span>
    </button>
  );
}
