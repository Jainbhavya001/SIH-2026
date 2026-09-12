import type { Verdict, Severity, TamperVerdict } from "@/types";

export function cx(...args: Array<string | false | null | undefined>): string {
  return args.filter(Boolean).join(" ");
}

export const verdictColors: Record<Verdict, { text: string; bg: string; ring: string; glow: string }> = {
  CLEAR: { text: "text-accent-emerald", bg: "bg-accent-emerald/10", ring: "ring-accent-emerald/30", glow: "shadow-glow-emerald" },
  REVIEW: { text: "text-accent-amber", bg: "bg-accent-amber/10", ring: "ring-accent-amber/30", glow: "shadow-glow" },
  REJECT: { text: "text-accent-rose", bg: "bg-accent-rose/10", ring: "ring-accent-rose/30", glow: "shadow-glow-rose" },
};

export const severityColors: Record<Severity, { text: string; bg: string; dot: string }> = {
  info: { text: "text-sky-300", bg: "bg-sky-400/10", dot: "bg-sky-400" },
  warning: { text: "text-accent-amber", bg: "bg-accent-amber/10", dot: "bg-accent-amber" },
  critical: { text: "text-accent-rose", bg: "bg-accent-rose/10", dot: "bg-accent-rose" },
};

export const tamperColors: Record<TamperVerdict, { text: string; bg: string }> = {
  clean: { text: "text-accent-emerald", bg: "bg-accent-emerald/10" },
  suspicious: { text: "text-accent-amber", bg: "bg-accent-amber/10" },
  tampered: { text: "text-accent-rose", bg: "bg-accent-rose/10" },
};

export function formatDocType(type: string): string {
  return type
    .split("_")
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

export function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatPercent(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return `${Math.round(v * 100)}%`;
}
