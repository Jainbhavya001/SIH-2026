import { motion } from "framer-motion";
import type { Verdict } from "@/types";

const VERDICT_STROKE: Record<Verdict, string> = {
  CLEAR: "#34d399",
  REVIEW: "#f59e0b",
  REJECT: "#fb7185",
};

const VERDICT_LABEL: Record<Verdict, string> = {
  CLEAR: "Clear to proceed",
  REVIEW: "Manual review required",
  REJECT: "Recommend rejection",
};

export function RiskGauge({ score, verdict }: { score: number; verdict: Verdict }) {
  const size = 220;
  const stroke = 14;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  // Sweep 270deg (like a speedometer) starting bottom-left, but a clean
  // full-circle progress ring reads better at a glance for a risk score.
  const offset = circumference - (score / 100) * circumference;
  const color = VERDICT_STROKE[verdict];

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={stroke}
          />
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
            style={{ filter: `drop-shadow(0 0 10px ${color}66)` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            className="text-5xl font-extrabold tabular-nums text-white"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            {score}
          </motion.span>
          <span className="text-xs font-medium uppercase tracking-widest text-slate-500">
            risk score
          </span>
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="mt-4 flex flex-col items-center gap-1"
      >
        <span
          className="rounded-full px-4 py-1.5 text-sm font-bold tracking-wide"
          style={{ backgroundColor: `${color}1a`, color, boxShadow: `0 0 20px ${color}22` }}
        >
          {verdict}
        </span>
        <span className="text-xs text-slate-400">{VERDICT_LABEL[verdict]}</span>
      </motion.div>
    </div>
  );
}
