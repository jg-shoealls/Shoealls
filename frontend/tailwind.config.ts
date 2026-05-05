import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        bg:        "var(--color-bg)",
        surface:   "var(--color-surface)",
        card:      "var(--color-card)",
        border:    "var(--color-border)",
        blue:      "#3B82F6",
        green:     "#10B981",
        amber:     "#F59E0B",
        red:       "#EF4444",
        purple:    "#AF65FA",
        textPri:   "var(--color-text-pri)",
        textSec:   "var(--color-text-sec)",
        textMuted: "var(--color-text-muted)",
      },
    },
  },
  plugins: [],
};
export default config;
