import { PropsWithChildren } from "react";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";

export function AppShell({
  title,
  subtitle,
  children,
}: PropsWithChildren<{ title: string; subtitle?: string }>) {
  return (
    <div className="relative flex min-h-screen bg-base-950 text-slate-100">
      <div className="pointer-events-none fixed inset-0 bg-grid bg-grid opacity-40" />
      <div className="pointer-events-none fixed inset-0 bg-radial-fade" />

      <Sidebar />

      <div className="relative z-10 flex min-h-screen flex-1 flex-col">
        <Topbar title={title} subtitle={subtitle} />
        <main className="flex-1 px-6 py-8">{children}</main>
      </div>
    </div>
  );
}
