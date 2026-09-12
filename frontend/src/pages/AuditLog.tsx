import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ListTree, RefreshCw } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Card, CardBody } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { RiskBadge } from "@/components/risk/RiskBadge";
import { Spinner } from "@/components/ui/Spinner";
import { listScans } from "@/lib/api";
import { formatDocType, formatTimestamp } from "@/lib/utils";
import type { ScanListItem } from "@/types";

export default function AuditLog() {
  const [scans, setScans] = useState<ScanListItem[] | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  function load() {
    setLoading(true);
    listScans()
      .then(setScans)
      .finally(() => setLoading(false));
  }

  useEffect(load, []);

  return (
    <AppShell title="Audit Log" subtitle="Digital trail of every screening decision">
      <Card>
        <div className="flex items-center justify-between border-b border-white/8 px-5 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white/5 text-accent-cyan">
              <ListTree size={17} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-100">Scan History</h3>
              <p className="text-xs text-slate-400">{scans?.length ?? 0} record(s) recorded this session</p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={load}>
            <RefreshCw size={14} /> Refresh
          </Button>
        </div>

        <CardBody className="p-0">
          {loading ? (
            <div className="flex items-center justify-center gap-2 py-16 text-slate-400">
              <Spinner /> Loading records…
            </div>
          ) : !scans || scans.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-16 text-center text-slate-500">
              <p>No scans recorded yet.</p>
              <Button variant="secondary" size="sm" onClick={() => navigate("/scan")}>
                Run your first screening
              </Button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-white/8 text-[11px] uppercase tracking-wide text-slate-500">
                    <th className="px-5 py-3">Scan ID</th>
                    <th className="px-5 py-3">Document Type</th>
                    <th className="px-5 py-3">Timestamp</th>
                    <th className="px-5 py-3">Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {scans.map((s, i) => (
                    <motion.tr
                      key={s.id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.02 }}
                      onClick={() => navigate(`/results/${s.id}`)}
                      className="cursor-pointer border-b border-white/5 transition-colors hover:bg-white/5"
                    >
                      <td className="px-5 py-3 font-mono text-xs text-slate-400">{s.id.slice(0, 12)}…</td>
                      <td className="px-5 py-3 text-slate-200">{formatDocType(s.document_type)}</td>
                      <td className="px-5 py-3 text-slate-400">{formatTimestamp(s.timestamp)}</td>
                      <td className="px-5 py-3">
                        <RiskBadge verdict={s.verdict} score={s.risk_score} />
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardBody>
      </Card>
    </AppShell>
  );
}
