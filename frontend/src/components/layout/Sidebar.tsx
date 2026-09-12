import { NavLink } from "react-router-dom";
import { LayoutDashboard, ScanLine, ListTree, ShieldCheck } from "lucide-react";
import { cx } from "@/lib/utils";

const navItems = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/scan", label: "New Scan", icon: ScanLine },
  { to: "/audit-log", label: "Audit Log", icon: ListTree },
];

export function Sidebar() {
  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r border-white/8 bg-base-900/60 px-4 py-6 lg:flex">
      <div className="mb-8 flex items-center gap-2.5 px-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-accent-cyan to-accent-teal text-base-950 shadow-glow">
          <ShieldCheck size={20} strokeWidth={2.4} />
        </div>
        <div>
          <p className="text-sm font-bold leading-none text-white">Sentinel</p>
          <p className="text-[11px] leading-none text-slate-400">Document Screening</p>
        </div>
      </div>

      <nav className="flex flex-1 flex-col gap-1">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              cx(
                "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors",
                isActive
                  ? "bg-white/8 text-white shadow-[inset_0_0_0_1px_rgba(255,255,255,0.08)]"
                  : "text-slate-400 hover:bg-white/5 hover:text-slate-100"
              )
            }
          >
            <Icon size={17} strokeWidth={2} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="rounded-xl border border-white/8 bg-white/5 px-3 py-3">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
          Problem Statement
        </p>
        <p className="mt-1 text-xs text-slate-300">SIH&nbsp;26188 &middot; SSB, MHA</p>
        <p className="mt-0.5 text-[11px] text-slate-500">Blockchain &amp; Cybersecurity</p>
      </div>
    </aside>
  );
}
