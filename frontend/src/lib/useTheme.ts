"use client";

import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "light";

export function useTheme() {
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    const stored = (localStorage.getItem("shoealls_theme") as Theme) ?? "dark";
    setTheme(stored);
    apply(stored);
  }, []);

  const toggle = useCallback(() => {
    setTheme((prev) => {
      const next: Theme = prev === "dark" ? "light" : "dark";
      localStorage.setItem("shoealls_theme", next);
      apply(next);
      return next;
    });
  }, []);

  return { theme, toggle };
}

function apply(t: Theme) {
  document.documentElement.classList.toggle("light", t === "light");
}
