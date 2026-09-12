import { PropsWithChildren } from "react";
import { cx } from "@/lib/utils";

export function Badge({
  children,
  className,
  dot,
}: PropsWithChildren<{ className?: string; dot?: string }>) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide",
        className
      )}
    >
      {dot && <span className={cx("h-1.5 w-1.5 rounded-full", dot)} />}
      {children}
    </span>
  );
}
