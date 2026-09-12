import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { ScanLine, ScanText, ShieldCheck, FileWarning, ScanFace, AlertTriangle } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { DocumentTypeSelect } from "@/components/upload/DocumentTypeSelect";
import { DocumentDropzone } from "@/components/upload/DocumentDropzone";
import { scanDocument } from "@/lib/api";
import type { DocumentType } from "@/types";

const STAGES = [
  { icon: ScanText, label: "Extracting text (OCR)" },
  { icon: ShieldCheck, label: "Validating document standards" },
  { icon: FileWarning, label: "Analyzing for tampering" },
  { icon: ScanFace, label: "Verifying face match" },
];

export default function ScanDashboard() {
  const [documentType, setDocumentType] = useState<DocumentType>("passport");
  const [document, setDocument] = useState<File | null>(null);
  const [liveFace, setLiveFace] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function handleSubmit() {
    if (!document) return;
    setLoading(true);
    setError(null);
    setStage(0);

    const interval = setInterval(() => {
      setStage((s) => Math.min(s + 1, STAGES.length - 1));
    }, 550);

    try {
      const result = await scanDocument({ documentType, document, liveFace });
      clearInterval(interval);
      navigate(`/results/${result.id}`, { state: result });
    } catch (e) {
      clearInterval(interval);
      setError(e instanceof Error ? e.message : "Scan failed.");
      setLoading(false);
    }
  }

  return (
    <AppShell title="New Screening" subtitle="Upload a document to run the full AI pipeline">
      <div className="mx-auto max-w-3xl">
        <Card>
          <CardHeader
            title="1. Document Type"
            subtitle="Select the kind of document being presented"
            icon={<ScanLine size={17} />}
          />
          <CardBody>
            <DocumentTypeSelect value={documentType} onChange={setDocumentType} />
          </CardBody>
        </Card>

        <Card className="mt-5">
          <CardHeader
            title="2. Upload Evidence"
            subtitle="A clear, well-lit photo works best for OCR and tamper analysis"
            icon={<ScanText size={17} />}
          />
          <CardBody className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <DocumentDropzone
              label="Document photo"
              hint="JPG / PNG — front page with MRZ if applicable"
              file={document}
              onFile={setDocument}
            />
            <DocumentDropzone
              label="Live capture (selfie)"
              hint="For face verification against the document photo"
              file={liveFace}
              onFile={setLiveFace}
              optional
            />
          </CardBody>
        </Card>

        {error && (
          <div className="mt-5 flex items-center gap-2.5 rounded-xl border border-accent-rose/25 bg-accent-rose/10 px-4 py-3 text-sm text-accent-rose">
            <AlertTriangle size={16} />
            {error}
          </div>
        )}

        <div className="mt-6 flex justify-end">
          <Button size="lg" disabled={!document || loading} onClick={handleSubmit}>
            {loading ? "Screening…" : "Run Screening"}
          </Button>
        </div>
      </div>

      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-base-950/80 backdrop-blur-sm"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="w-full max-w-sm rounded-2xl border border-white/10 bg-base-900 p-6 shadow-card"
            >
              <div className="relative mx-auto mb-5 flex h-20 w-20 items-center justify-center">
                <span className="absolute inset-0 rounded-full border-2 border-accent-cyan/40 animate-pulse-ring" />
                <span className="absolute inset-0 rounded-full border-2 border-accent-cyan/30" />
                <ScanLine size={30} className="text-accent-cyan" />
              </div>
              <div className="space-y-3">
                {STAGES.map((s, i) => (
                  <div key={s.label} className="flex items-center gap-3">
                    <span
                      className={`flex h-7 w-7 items-center justify-center rounded-full border text-[11px] font-bold ${
                        i < stage
                          ? "border-accent-emerald/40 bg-accent-emerald/15 text-accent-emerald"
                          : i === stage
                          ? "border-accent-cyan/40 bg-accent-cyan/15 text-accent-cyan"
                          : "border-white/10 bg-white/5 text-slate-500"
                      }`}
                    >
                      {i < stage ? "✓" : i + 1}
                    </span>
                    <span className={`text-sm ${i <= stage ? "text-slate-200" : "text-slate-500"}`}>{s.label}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </AppShell>
  );
}
