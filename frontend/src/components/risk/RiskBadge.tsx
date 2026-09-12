import type { Verdict } from "@/types";
import { verdictColors } from "@/lib/utils";
import { Badge } from "@/components/ui/Badge";

export function RiskBadge({ verdict, score }: { verdict: Verdict; score?: number }) {
  const c = verdictColors[verdict];
  return (
    <Badge className={`${c.text} ${c.bg} ring-1 ${c.ring}`}>
      {verdict}
      {typeof score === "number" && <span className="opacity-70">&middot; {score}</span>}
    </Badge>
  );
}
