import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: 'class',
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        bg:          "var(--color-bg)",
        surface:     "var(--color-surface)",
        card:        "var(--color-card)",
        border:      "var(--color-border)",
        primaryBlue: "#2563EB",   // Tailwind Blue 600
        medicalTeal: "#0D9488",   // Tailwind Teal 600
        accentIndigo:"#4F46E5",   // Tailwind Indigo 600
        warningAmber:"#D97706",   // Tailwind Amber 600
        dangerRed:   "#DC2626",   // Tailwind Red 600
        // 이전 페이지 호환성을 위한 별칭 복구
        blue:        "#2563EB",
        emerald:     "#0D9488",
        purple:      "#4F46E5",
        amber:       "#D97706",
        red:         "#DC2626",
        neonBlue:    "#2563EB",
        textPri:     "var(--color-text-pri)",
        textSec:     "var(--color-text-sec)",
        textMuted:   "var(--color-text-muted)",
      },
      backgroundImage: {
        'mesh': 'radial-gradient(at 40% 20%, var(--color-mesh-1) 0px, transparent 50%), radial-gradient(at 80% 0%, var(--color-mesh-2) 0px, transparent 50%), radial-gradient(at 0% 50%, var(--color-mesh-3) 0px, transparent 50%)',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out forwards',
        'fade-in-up': 'fadeInUp 0.6s ease-out forwards',
        'spin-slow': 'spin 3s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(15px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        }
      }
    },
  },
  plugins: [],
};
export default config;
