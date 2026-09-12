"""
API-facing Pydantic models (the wire contract with the frontend). Kept
separate from the domain dataclasses in each module so internal refactors
don't silently change the API shape.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

DocumentTypeLiteral = Literal["passport", "visa", "national_id", "driving_license", "permit"]
VerdictLiteral = Literal["CLEAR", "REVIEW", "REJECT"]


class OCRSummary(BaseModel):
    raw_text: str
    mean_confidence: float
    engine_available: bool
    warning: str | None = None
    extracted_fields: dict[str, Any] = {}


class MRZField(BaseModel):
    name: str
    value: str
    valid: bool | None


class MRZSummary(BaseModel):
    detected: bool
    format: str | None = None
    issuing_country: str | None = None
    surname: str | None = None
    given_names: str | None = None
    nationality: str | None = None
    sex: str | None = None
    date_of_birth: str | None = None
    date_of_expiry: str | None = None
    composite_valid: bool | None = None
    fields: list[MRZField] = []
    warnings: list[str] = []


class ValidationIssueModel(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "critical"]
    field: str | None = None


class ValidationSummary(BaseModel):
    score: int
    issues: list[ValidationIssueModel]
    checks_run: int


class TamperingSummary(BaseModel):
    tampering_score: int
    verdict: Literal["clean", "suspicious", "tampered"]
    evidence: list[str]
    ela_suspicious: bool
    ela_max_error: float
    copy_move_matches: int
    ela_heatmap: str | None = None  # data: URL, present only if generated


class FaceSummary(BaseModel):
    attempted: bool
    similarity: float | None = None
    is_match: bool | None = None
    backend: str | None = None
    document_face_found: bool = False
    live_face_found: bool = False


class RiskSummary(BaseModel):
    risk_score: int
    verdict: VerdictLiteral
    contributing_factors: list[str]
    validation_score: int
    tampering_score: int
    face_similarity: float | None = None


class ScanResponse(BaseModel):
    id: str
    timestamp: str
    document_type: DocumentTypeLiteral
    ocr: OCRSummary
    mrz: MRZSummary | None = None
    validation: ValidationSummary
    tampering: TamperingSummary
    face: FaceSummary
    risk: RiskSummary


class ScanListItem(BaseModel):
    id: str
    timestamp: str
    document_type: DocumentTypeLiteral
    risk_score: int
    verdict: VerdictLiteral


class StatsResponse(BaseModel):
    total_scans: int
    by_verdict: dict[str, int]
