import { motion } from "framer-motion";
import { cx } from "@/lib/utils";

/**
 * `invert` = true means "higher is better" (e.g. validation score, face
 * similarity), so the bar renders green-at-high. Default is "higher is
 * worse" (tampering score), rendering red-at-high.
 */
export function ScoreBar({
  label,
  value,
  max = 100,
  invert = false,
  suffix,
}: {
  label: string;
  value: number;
  max?: number;
  invert?: boolean;
  suffix?: string;
}) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  const goodness = invert ? pct : 100 - pct;

  const color =
    goodness >= 70 ? "#34d399" : goodness >= 40 ? "#f59e0b" : "#fb7185";

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-xs font-medium text-slate-300">{label}</span>
        <span className="text-xs font-semibold tabular-nums text-slate-200">
          {value}
          {suffix ?? `/${max}`}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-white/8">
        <motion.div
          className={cx("h-full rounded-full")}
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
    </div>
  );
}
