import { useEffect, useState } from "react";
import { Radio, WifiOff } from "lucide-react";
import { getHealth } from "@/lib/api";
import type { HealthResponse } from "@/types";

export function Topbar({ title, subtitle }: { title: string; subtitle?: string }) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let mounted = true;
    getHealth()
      .then((h) => mounted && setHealth(h))
      .catch(() => mounted && setError(true));
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <header className="flex items-center justify-between border-b border-white/8 bg-base-950/70 px-6 py-5 backdrop-blur-md">
      <div>
        <h1 className="text-lg font-bold text-white">{title}</h1>
        {subtitle && <p className="mt-0.5 text-sm text-slate-400">{subtitle}</p>}
      </div>

      <div className="flex items-center gap-3">
        {error ? (
          <StatusPill icon={<WifiOff size={13} />} label="API offline" tone="rose" />
        ) : health ? (
          <>
            <StatusPill
              icon={<Radio size={13} />}
              label={`OCR: ${health.engines.ocr === "unavailable" ? "offline" : "live"}`}
              tone={health.engines.ocr === "unavailable" ? "amber" : "emerald"}
            />
            <StatusPill
              icon={<Radio size={13} />}
              label={`Face: ${health.engines.face_verification.includes("dlib") ? "embeddings" : "lightweight"}`}
              tone="cyan"
            />
          </>
        ) : (
          <StatusPill icon={<Radio size={13} className="animate-pulse" />} label="Checking…" tone="slate" />
        )}
      </div>
    </header>
  );
}

function StatusPill({
  icon,
  label,
  tone,
}: {
  icon: React.ReactNode;
  label: string;
  tone: "emerald" | "amber" | "rose" | "cyan" | "slate";
}) {
  const toneClasses: Record<string, string> = {
    emerald: "bg-accent-emerald/10 text-accent-emerald border-accent-emerald/25",
    amber: "bg-accent-amber/10 text-accent-amber border-accent-amber/25",
    rose: "bg-accent-rose/10 text-accent-rose border-accent-rose/25",
    cyan: "bg-accent-cyan/10 text-accent-cyan border-accent-cyan/25",
    slate: "bg-white/5 text-slate-400 border-white/10",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${toneClasses[tone]}`}
    >
      {icon}
      {label}
    </span>
  );
}
