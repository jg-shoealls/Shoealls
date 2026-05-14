import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        bg:        "rgb(var(--bg) / <alpha-value>)",
        surface:   "rgb(var(--surface) / <alpha-value>)",
        card:      "rgb(var(--card) / <alpha-value>)",
        border:    "rgb(var(--border) / <alpha-value>)",
        blue:      "rgb(var(--blue) / <alpha-value>)",
        green:     "rgb(var(--green) / <alpha-value>)",
        amber:     "rgb(var(--amber) / <alpha-value>)",
        red:       "rgb(var(--red) / <alpha-value>)",
        purple:    "rgb(var(--purple) / <alpha-value>)",
        textPri:   "rgb(var(--textPri) / <alpha-value>)",
        textSec:   "rgb(var(--textSec) / <alpha-value>)",
        textMuted: "rgb(var(--textMuted) / <alpha-value>)",
      },
    },
  },
  plugins: [],
};
export default config;
