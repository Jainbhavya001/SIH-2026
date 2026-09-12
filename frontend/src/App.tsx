import { Routes, Route } from "react-router-dom";
import LandingPage from "@/pages/LandingPage";
import ScanDashboard from "@/pages/ScanDashboard";
import ScanResult from "@/pages/ScanResult";
import AuditLog from "@/pages/AuditLog";
import NotFound from "@/pages/NotFound";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/scan" element={<ScanDashboard />} />
      <Route path="/results/:id" element={<ScanResult />} />
      <Route path="/audit-log" element={<AuditLog />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}
