import { useEffect, useState } from "react";
import { useLocation, useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ScanText,
  ShieldCheck,
  FileWarning,
  ScanFace,
  CheckCircle2,
  XCircle,
  HelpCircle,
  ArrowLeft,
  AlertTriangle,
} from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { RiskGauge } from "@/components/risk/RiskGauge";
import { ScoreBar } from "@/components/risk/ScoreBar";
import { getScan } from "@/lib/api";
import { severityColors, tamperColors, formatDocType, formatTimestamp } from "@/lib/utils";
import type { ScanResponse } from "@/types";

export default function ScanResult() {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const navigate = useNavigate();
  const preloaded = location.state as ScanResponse | undefined;

  const [data, setData] = useState<ScanResponse | null>(preloaded && preloaded.id === id ? preloaded : null);
  const [loading, setLoading] = useState(!data);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (data || !id) return;
    getScan(id)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [id, data]);

  if (loading) {
    return (
      <AppShell title="Screening Report">
        <div className="flex h-64 items-center justify-center gap-3 text-slate-400">
          <Spinner /> Loading report…
        </div>
      </AppShell>
    );
  }

  if (error || !data) {
    return (
      <AppShell title="Screening Report">
        <div className="flex flex-col items-center gap-3 py-16 text-center text-slate-400">
          <AlertTriangle className="text-accent-rose" />
          <p>{error ?? "Report not found."}</p>
          <Button variant="secondary" onClick={() => navigate("/scan")}>
            Start a new screening
          </Button>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell
      title="Screening Report"
      subtitle={`${formatDocType(data.document_type)} · Scan ${data.id.slice(0, 8)} · ${formatTimestamp(data.timestamp)}`}
    >
      <button
        onClick={() => navigate(-1)}
        className="mb-5 flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200"
      >
        <ArrowLeft size={14} /> Back
      </button>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        {/* Left: Risk summary */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="lg:col-span-1">
          <Card>
            <CardBody className="flex flex-col items-center py-8">
              <RiskGauge score={data.risk.risk_score} verdict={data.risk.verdict} />
            </CardBody>
          </Card>

          <Card className="mt-5">
            <CardHeader title="Module Scores" icon={<ShieldCheck size={16} />} />
            <CardBody className="space-y-4">
              <ScoreBar label="Validation" value={data.risk.validation_score} invert />
              <ScoreBar label="Tampering risk" value={data.risk.tampering_score} />
              {data.face.attempted && data.risk.face_similarity !== null && (
                <ScoreBar
                  label="Face similarity"
                  value={Math.round(data.risk.face_similarity * 100)}
                  invert
                  suffix="%"
                />
              )}
            </CardBody>
          </Card>

          <Card className="mt-5">
            <CardHeader title="Contributing Factors" icon={<AlertTriangle size={16} />} />
            <CardBody>
              <ul className="space-y-2.5">
                {data.risk.contributing_factors.map((f, i) => (
                  <li key={i} className="flex gap-2 text-xs leading-relaxed text-slate-300">
                    <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-accent-cyan" />
                    {f}
                  </li>
                ))}
              </ul>
            </CardBody>
          </Card>
        </motion.div>

        {/* Right: module detail */}
        <div className="space-y-5 lg:col-span-2">
          <OCRSection data={data} />
          {data.mrz && <MRZSection data={data} />}
          <ValidationSection data={data} />
          <TamperingSection data={data} />
          <FaceSection data={data} />
        </div>
      </div>
    </AppShell>
  );
}

function Section({
  title,
  icon,
  action,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
      <Card>
        <CardHeader title={title} icon={icon} action={action} />
        <CardBody>{children}</CardBody>
      </Card>
    </motion.div>
  );
}

function OCRSection({ data }: { data: ScanResponse }) {
  const fields = Object.entries(data.ocr.extracted_fields ?? {});
  return (
    <Section title="Module 1 · OCR Extraction" icon={<ScanText size={16} />}>
      {!data.ocr.engine_available && (
        <p className="mb-3 rounded-lg border border-accent-amber/25 bg-accent-amber/10 px-3 py-2 text-xs text-accent-amber">
          {data.ocr.warning}
        </p>
      )}
      {fields.length > 0 ? (
        <dl className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
          {fields.map(([k, v]) => (
            <div key={k} className="rounded-lg bg-white/5 px-3 py-2">
              <dt className="text-[11px] uppercase tracking-wide text-slate-500">{k.replace(/_/g, " ")}</dt>
              <dd className="mt-0.5 font-mono text-sm text-slate-200">{v || "—"}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="text-xs text-slate-500">
          {data.mrz?.detected
            ? "No extra visual-zone fields extracted — see MRZ breakdown below."
            : "No fields could be extracted from this document."}
        </p>
      )}
      <p className="mt-3 text-[11px] text-slate-500">OCR confidence: {data.ocr.mean_confidence.toFixed(1)}%</p>
    </Section>
  );
}

function CheckIcon({ valid }: { valid: boolean | null }) {
  if (valid === null) return <HelpCircle size={14} className="text-slate-500" />;
  return valid ? (
    <CheckCircle2 size={14} className="text-accent-emerald" />
  ) : (
    <XCircle size={14} className="text-accent-rose" />
  );
}

function MRZSection({ data }: { data: ScanResponse }) {
  const mrz = data.mrz!;
  return (
    <Section title="Machine-Readable Zone (ICAO 9303)" icon={<ShieldCheck size={16} />}>
      {!mrz.detected ? (
        <p className="text-xs text-slate-500">{mrz.warnings.join(" ")}</p>
      ) : (
        <>
          <div className="mb-4 grid grid-cols-2 gap-2.5 sm:grid-cols-3">
            <InfoTile label="Country" value={mrz.issuing_country} />
            <InfoTile label="Name" value={[mrz.given_names, mrz.surname].filter(Boolean).join(" ")} />
            <InfoTile label="Nationality" value={mrz.nationality} />
            <InfoTile label="Sex" value={mrz.sex} />
            <InfoTile label="Date of Birth" value={mrz.date_of_birth} />
            <InfoTile label="Date of Expiry" value={mrz.date_of_expiry} />
          </div>
          <div className="space-y-1.5">
            {mrz.fields.map((f) => (
              <div key={f.name} className="flex items-center justify-between rounded-lg bg-white/5 px-3 py-2">
                <span className="text-xs text-slate-300">{f.name.replace(/_/g, " ")} check digit</span>
                <span className="flex items-center gap-1.5 font-mono text-xs text-slate-400">
                  {f.value}
                  <CheckIcon valid={f.valid} />
                </span>
              </div>
            ))}
            <div className="flex items-center justify-between rounded-lg bg-white/5 px-3 py-2">
              <span className="text-xs font-semibold text-slate-200">Composite (overall) checksum</span>
              <CheckIcon valid={mrz.composite_valid} />
            </div>
          </div>
        </>
      )}
    </Section>
  );
}

function InfoTile({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <div className="rounded-lg bg-white/5 px-3 py-2">
      <p className="text-[11px] uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-0.5 font-mono text-sm text-slate-200">{value || "—"}</p>
    </div>
  );
}

function ValidationSection({ data }: { data: ScanResponse }) {
  const v = data.validation;
  return (
    <Section
      title="Module 2 · Document Validation"
      icon={<ShieldCheck size={16} />}
      action={<span className="text-xs font-semibold text-slate-400">Score {v.score}/100</span>}
    >
      {v.issues.length === 0 ? (
        <p className="text-xs text-accent-emerald">All {v.checks_run} rule checks passed.</p>
      ) : (
        <ul className="space-y-2">
          {v.issues.map((issue) => {
            const c = severityColors[issue.severity];
            return (
              <li key={issue.code} className={`rounded-lg px-3 py-2.5 ${c.bg}`}>
                <div className="flex items-center gap-2">
                  <span className={`h-1.5 w-1.5 rounded-full ${c.dot}`} />
                  <span className={`text-[11px] font-bold uppercase tracking-wide ${c.text}`}>{issue.severity}</span>
                  <span className="text-[11px] text-slate-500">{issue.code}</span>
                </div>
                <p className="mt-1 text-xs text-slate-300">{issue.message}</p>
              </li>
            );
          })}
        </ul>
      )}
    </Section>
  );
}

function TamperingSection({ data }: { data: ScanResponse }) {
  const t = data.tampering;
  const c = tamperColors[t.verdict];
  return (
    <Section
      title="Module 3 · Tampering Detection"
      icon={<FileWarning size={16} />}
      action={
        <span className={`rounded-full px-2.5 py-1 text-[11px] font-bold uppercase ${c.bg} ${c.text}`}>
          {t.verdict}
        </span>
      }
    >
      <div className="mb-4 grid grid-cols-2 gap-2.5 sm:grid-cols-3">
        <InfoTile label="ELA Max Error" value={t.ela_max_error.toFixed(0)} />
        <InfoTile label="Copy-move matches" value={String(t.copy_move_matches)} />
        <InfoTile label="Tampering score" value={`${t.tampering_score}/100`} />
      </div>

      <ul className="mb-4 space-y-1.5">
        {t.evidence.map((e, i) => (
          <li key={i} className="flex gap-2 text-xs leading-relaxed text-slate-300">
            <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-accent-amber" />
            {e}
          </li>
        ))}
      </ul>

      {t.ela_heatmap && (
        <div>
          <p className="mb-1.5 text-[11px] uppercase tracking-wide text-slate-500">
            Error Level Analysis heatmap (brighter = higher compression-error anomaly)
          </p>
          <img src={t.ela_heatmap} alt="ELA heatmap" className="w-full rounded-lg border border-white/10" />
        </div>
      )}
    </Section>
  );
}

function FaceSection({ data }: { data: ScanResponse }) {
  const f = data.face;
  return (
    <Section title="Module 4 · Face Verification" icon={<ScanFace size={16} />}>
      {!f.attempted ? (
        <p className="text-xs text-slate-500">
          No live capture was provided for this scan — face verification was skipped.
          {f.document_face_found ? " A face was detected on the document photo." : " No face was detected on the document photo."}
        </p>
      ) : (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2.5">
            <InfoTile label="Document face" value={f.document_face_found ? "Detected" : "Not found"} />
            <InfoTile label="Live face" value={f.live_face_found ? "Detected" : "Not found"} />
          </div>
          <ScoreBar label="Similarity" value={Math.round((f.similarity ?? 0) * 100)} invert suffix="%" />
          <p className={`text-xs font-semibold ${f.is_match ? "text-accent-emerald" : "text-accent-rose"}`}>
            {f.is_match ? "Match — same individual likely" : "No match — presenter may differ from document"}
          </p>
          <p className="text-[11px] text-slate-500">Backend: {f.backend}</p>
        </div>
      )}
    </Section>
  );
}
