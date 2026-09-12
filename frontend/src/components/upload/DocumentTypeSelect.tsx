import { BookText, Plane, IdCard, Car, FileCheck2 } from "lucide-react";
import type { DocumentType } from "@/types";
import { cx } from "@/lib/utils";

const OPTIONS: { value: DocumentType; label: string; hint: string; icon: React.ReactNode }[] = [
  { value: "passport", label: "Passport", hint: "MRZ checksum verified", icon: <BookText size={18} /> },
  { value: "visa", label: "Visa", hint: "Entry & stay rules", icon: <Plane size={18} /> },
  { value: "national_id", label: "National ID", hint: "Identity card", icon: <IdCard size={18} /> },
  { value: "driving_license", label: "Driving Licence", hint: "Permit to drive", icon: <Car size={18} /> },
  { value: "permit", label: "Permit", hint: "Travel authorization", icon: <FileCheck2 size={18} /> },
];

export function DocumentTypeSelect({
  value,
  onChange,
}: {
  value: DocumentType;
  onChange: (v: DocumentType) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-5">
      {OPTIONS.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={cx(
              "flex flex-col items-start gap-2 rounded-xl border px-3.5 py-3 text-left transition-all",
              active
                ? "border-accent-cyan/40 bg-accent-cyan/10 shadow-glow"
                : "border-white/10 bg-white/5 hover:bg-white/8"
            )}
          >
            <span className={cx("flex h-8 w-8 items-center justify-center rounded-lg", active ? "bg-accent-cyan/20 text-accent-cyan" : "bg-white/8 text-slate-300")}>
              {opt.icon}
            </span>
            <div>
              <p className={cx("text-sm font-semibold", active ? "text-white" : "text-slate-200")}>{opt.label}</p>
              <p className="text-[11px] text-slate-500">{opt.hint}</p>
            </div>
          </button>
        );
      })}
    </div>
  );
}
