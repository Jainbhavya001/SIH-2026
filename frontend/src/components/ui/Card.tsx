import { PropsWithChildren } from "react";
import { cx } from "@/lib/utils";

export function Card({
  children,
  className,
  glow = false,
}: PropsWithChildren<{ className?: string; glow?: boolean }>) {
  return (
    <div
      className={cx(
        "glass rounded-2xl border border-white/10 shadow-card",
        glow && "shadow-glow",
        className
      )}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  icon,
  action,
}: {
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-white/8 px-5 py-4">
      <div className="flex items-start gap-3">
        {icon && (
          <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white/5 text-accent-cyan">
            {icon}
          </div>
        )}
        <div>
          <h3 className="text-sm font-semibold tracking-wide text-slate-100">{title}</h3>
          {subtitle && <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>}
        </div>
      </div>
      {action}
    </div>
  );
}

export function CardBody({ children, className }: PropsWithChildren<{ className?: string }>) {
  return <div className={cx("px-5 py-4", className)}>{children}</div>;
}
