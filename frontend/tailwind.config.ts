import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        base: {
          950: "#060a12",
          900: "#0a0f1a",
          850: "#0d1424",
          800: "#111a2e",
          700: "#1a2540",
          600: "#26355a",
          500: "#3a4d7a",
        },
        accent: {
          cyan: "#22d3ee",
          teal: "#2dd4bf",
          amber: "#f59e0b",
          rose: "#fb7185",
          emerald: "#34d399",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(34,211,238,0.15), 0 0 24px rgba(34,211,238,0.12)",
        "glow-rose": "0 0 0 1px rgba(251,113,133,0.2), 0 0 24px rgba(251,113,133,0.15)",
        "glow-emerald": "0 0 0 1px rgba(52,211,153,0.2), 0 0 24px rgba(52,211,153,0.15)",
        card: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 8px 30px rgba(0,0,0,0.35)",
      },
      backgroundImage: {
        grid: "linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px)",
        "radial-fade": "radial-gradient(circle at 50% 0%, rgba(34,211,238,0.10), transparent 60%)",
      },
      backgroundSize: {
        grid: "28px 28px",
      },
      keyframes: {
        scan: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        "pulse-ring": {
          "0%": { transform: "scale(0.9)", opacity: "0.6" },
          "100%": { transform: "scale(1.4)", opacity: "0" },
        },
      },
      animation: {
        scan: "scan 2.4s linear infinite",
        "pulse-ring": "pulse-ring 1.8s cubic-bezier(0.2,0.6,0.4,1) infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
