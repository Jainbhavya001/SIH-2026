import { useCallback, useRef, useState } from "react";
import { UploadCloud, X, ImageIcon } from "lucide-react";
import { cx } from "@/lib/utils";

export function DocumentDropzone({
  label,
  hint,
  file,
  onFile,
  optional = false,
}: {
  label: string;
  hint?: string;
  file: File | null;
  onFile: (f: File | null) => void;
  optional?: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const previewUrl = file ? URL.createObjectURL(file) : null;

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files?.[0];
      if (f) onFile(f);
    },
    [onFile]
  );

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <label className="text-sm font-medium text-slate-200">{label}</label>
        {optional && <span className="text-[11px] text-slate-500">Optional</span>}
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={cx(
          "group relative flex min-h-[160px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed transition-colors",
          dragging
            ? "border-accent-cyan bg-accent-cyan/5"
            : "border-white/12 bg-white/[0.03] hover:border-white/25 hover:bg-white/[0.05]"
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => onFile(e.target.files?.[0] ?? null)}
        />

        {previewUrl ? (
          <>
            <img src={previewUrl} alt="preview" className="absolute inset-0 h-full w-full object-cover opacity-90" />
            <div className="absolute inset-0 bg-gradient-to-t from-base-950/85 via-base-950/10 to-transparent" />
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onFile(null);
              }}
              className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-full bg-base-950/80 text-slate-200 hover:bg-base-950"
            >
              <X size={14} />
            </button>
            <div className="absolute bottom-2 left-2 flex items-center gap-1.5 rounded-full bg-base-950/70 px-2.5 py-1 text-[11px] text-slate-200">
              <ImageIcon size={12} />
              {file?.name}
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center gap-2 px-4 text-center">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/6 text-accent-cyan transition-transform group-hover:scale-105">
              <UploadCloud size={20} />
            </span>
            <p className="text-sm font-medium text-slate-200">Drop image or click to browse</p>
            {hint && <p className="text-xs text-slate-500">{hint}</p>}
          </div>
        )}
      </div>
    </div>
  );
}
