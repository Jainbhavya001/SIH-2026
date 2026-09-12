import { Link } from "react-router-dom";
import { ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/Button";

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-base-950 text-center text-slate-300">
      <ShieldAlert size={40} className="text-accent-amber" />
      <h1 className="text-2xl font-bold text-white">404 — Page not found</h1>
      <p className="text-sm text-slate-500">The screen you're looking for doesn't exist.</p>
      <Link to="/">
        <Button variant="secondary">Back to Overview</Button>
      </Link>
    </div>
  );
}
