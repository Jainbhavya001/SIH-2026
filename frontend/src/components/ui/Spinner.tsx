import { cx } from "@/lib/utils";

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={cx(
        "inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-accent-cyan",
        className
      )}
    />
  );
}
