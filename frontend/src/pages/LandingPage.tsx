import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import {
  ScanText,
  ShieldCheck,
  FileWarning,
  ScanFace,
  ArrowRight,
  Gauge,
} from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Card, CardBody } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { getStats } from "@/lib/api";
import type { StatsResponse } from "@/types";

const MODULES = [
  {
    icon: ScanText,
    title: "OCR Extraction",
    desc: "Pulls structured fields from passports, visas, IDs, licences & permits — including the passport's machine-readable zone.",
  },
  {
    icon: ShieldCheck,
    title: "Document Validation",
    desc: "Runs ICAO 9303 checksum math, expiry/date-logic rules, and cross-field consistency checks.",
  },
  {
    icon: FileWarning,
    title: "Tampering Detection",
    desc: "Error Level Analysis, copy-move forensics, and metadata inspection catch photo swaps, stamp forgery, and edits.",
  },
  {
    icon: ScanFace,
    title: "Face Verification",
    desc: "Compares the document photo against a live capture to confirm the presenter's identity.",
  },
];

export default function LandingPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    getStats().then(setStats).catch(() => setStats({ total_scans: 0, by_verdict: { CLEAR: 0, REVIEW: 0, REJECT: 0 } }));
  }, []);

  return (
    <AppShell title="Command Overview" subtitle="Border checkpoint screening at a glance">
      {/* Hero */}
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <Card className="relative overflow-hidden">
          <div className="absolute -right-24 -top-24 h-72 w-72 rounded-full bg-accent-cyan/10 blur-3xl" />
          <div className="absolute -bottom-24 -left-10 h-64 w-64 rounded-full bg-accent-teal/10 blur-3xl" />
          <CardBody className="relative flex flex-col gap-6 px-8 py-10 lg:flex-row lg:items-center lg:justify-between">
            <div className="max-w-xl">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-accent-cyan/30 bg-accent-cyan/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-accent-cyan">
                SIH 26188 &middot; Ministry of Home Affairs &middot; SSB
              </span>
              <h2 className="mt-4 text-3xl font-extrabold leading-tight text-white lg:text-4xl">
                AI-Based Fake Identity &amp;{" "}
                <span className="text-gradient">Document Screening</span>
              </h2>
              <p className="mt-3 text-sm leading-relaxed text-slate-400">
                Screen passports, visas, IDs, licences, and permits in seconds — with
                machine-verifiable checksum validation, forensic tampering detection,
                and face verification behind one transparent risk score.
              </p>
              <Button size="lg" className="mt-6" onClick={() => navigate("/scan")}>
                Start a Screening <ArrowRight size={16} />
              </Button>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <StatTile label="Total Scans" value={stats?.total_scans ?? 0} tone="cyan" />
              <StatTile label="Clear" value={stats?.by_verdict?.CLEAR ?? 0} tone="emerald" />
              <StatTile label="Flagged" value={(stats?.by_verdict?.REVIEW ?? 0) + (stats?.by_verdict?.REJECT ?? 0)} tone="rose" />
            </div>
          </CardBody>
        </Card>
      </motion.div>

      {/* Module grid */}
      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {MODULES.map((m, i) => (
          <motion.div
            key={m.title}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 + i * 0.07 }}
          >
            <Card className="h-full">
              <CardBody>
                <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-accent-cyan/10 text-accent-cyan">
                  <m.icon size={19} />
                </div>
                <p className="text-sm font-semibold text-slate-100">
                  Module {i + 1} &middot; {m.title}
                </p>
                <p className="mt-1.5 text-xs leading-relaxed text-slate-400">{m.desc}</p>
              </CardBody>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Impact strip */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="mt-6">
        <Card>
          <CardBody className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-emerald/10 text-accent-emerald">
                <Gauge size={19} />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-100">Expected Impact</p>
                <p className="text-xs text-slate-400">
                  Minutes to seconds &middot; standardized decisions &middot; a full digital audit trail
                </p>
              </div>
            </div>
            <Button variant="secondary" onClick={() => navigate("/audit-log")}>
              View Audit Log
            </Button>
          </CardBody>
        </Card>
      </motion.div>
    </AppShell>
  );
}

function StatTile({ label, value, tone }: { label: string; value: number; tone: "cyan" | "emerald" | "rose" }) {
  const toneClass = { cyan: "text-accent-cyan", emerald: "text-accent-emerald", rose: "text-accent-rose" }[tone];
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-center">
      <p className={`text-2xl font-extrabold tabular-nums ${toneClass}`}>{value}</p>
      <p className="mt-0.5 text-[11px] uppercase tracking-wide text-slate-500">{label}</p>
    </div>
  );
}
