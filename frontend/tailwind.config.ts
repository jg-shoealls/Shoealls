import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        bg:        "#0F172A",
        surface:   "#1E2940",
        card:      "#263348",
        border:    "#32425B",
        blue:      "#3B82F6",
        green:     "#10B981",
        amber:     "#F59E0B",
        red:       "#EF4444",
        purple:    "#AF65FA",
        textPri:   "#F1F5F9",
        textSec:   "#94A3B8",
        textMuted: "#607085",
      },
    },
  },
  plugins: [],
};
export default config;
