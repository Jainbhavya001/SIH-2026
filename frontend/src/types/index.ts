export type DocumentType = "passport" | "visa" | "national_id" | "driving_license" | "permit";
export type Verdict = "CLEAR" | "REVIEW" | "REJECT";
export type Severity = "info" | "warning" | "critical";
export type TamperVerdict = "clean" | "suspicious" | "tampered";

export interface OCRSummary {
  raw_text: string;
  mean_confidence: number;
  engine_available: boolean;
  warning: string | null;
  extracted_fields: Record<string, string | null>;
}

export interface MRZField {
  name: string;
  value: string;
  valid: boolean | null;
}

export interface MRZSummary {
  detected: boolean;
  format: string | null;
  issuing_country: string | null;
  surname: string | null;
  given_names: string | null;
  nationality: string | null;
  sex: string | null;
  date_of_birth: string | null;
  date_of_expiry: string | null;
  composite_valid: boolean | null;
  fields: MRZField[];
  warnings: string[];
}

export interface ValidationIssue {
  code: string;
  message: string;
  severity: Severity;
  field: string | null;
}

export interface ValidationSummary {
  score: number;
  issues: ValidationIssue[];
  checks_run: number;
}

export interface TamperingSummary {
  tampering_score: number;
  verdict: TamperVerdict;
  evidence: string[];
  ela_suspicious: boolean;
  ela_max_error: number;
  copy_move_matches: number;
  ela_heatmap: string | null;
}

export interface FaceSummary {
  attempted: boolean;
  similarity: number | null;
  is_match: boolean | null;
  backend: string | null;
  document_face_found: boolean;
  live_face_found: boolean;
}

export interface RiskSummary {
  risk_score: number;
  verdict: Verdict;
  contributing_factors: string[];
  validation_score: number;
  tampering_score: number;
  face_similarity: number | null;
}

export interface ScanResponse {
  id: string;
  timestamp: string;
  document_type: DocumentType;
  ocr: OCRSummary;
  mrz: MRZSummary | null;
  validation: ValidationSummary;
  tampering: TamperingSummary;
  face: FaceSummary;
  risk: RiskSummary;
}

export interface ScanListItem {
  id: string;
  timestamp: string;
  document_type: DocumentType;
  risk_score: number;
  verdict: Verdict;
}

export interface StatsResponse {
  total_scans: number;
  by_verdict: Record<string, number>;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  engines: {
    ocr: string;
    face_verification: string;
  };
}
