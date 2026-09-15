"""
API-facing Pydantic models (the wire contract with the frontend). Kept
separate from the domain dataclasses in each module so internal refactors
don't silently change the API shape.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

DocumentTypeLiteral = Literal[
    "passport", "visa", "national_id", "driving_license", "permit",
    "pan_card", "aadhaar",
]
VerdictLiteral = Literal["CLEAR", "REVIEW", "REJECT"]


class OCRSummary(BaseModel):
    raw_text: str
    mean_confidence: float
    engine_available: bool
    warning: str | None = None
    extracted_fields: dict[str, Any] = {}
    # label -> text read from that region's crop (only when regions were localized)
    region_text: dict[str, str] = {}


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
    source: str | None = None  # "mrz_region" | "full_image"


class ValidationIssueModel(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "critical"]
    field: str | None = None


class ValidationSummary(BaseModel):
    score: int
    issues: list[ValidationIssueModel]
    checks_run: int


class BoundingBox(BaseModel):
    """Axis-aligned box in absolute pixel coordinates of the uploaded image."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height


class DetectedRegion(BaseModel):
    label: str
    bbox: BoundingBox
    cropped_image_path: str | None = None


class RegionLocalizationResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    regions: list[DetectedRegion] = []
    processing_time_ms: float = 0.0
    model_version: str
    # "faster_rcnn" when a trained model ran; "heuristic" / "full_image" are
    # the CPU fallbacks; "disabled" when localization is switched off.
    backend: str = "faster_rcnn"
    fallback_used: bool = False
    image_width: int | None = None
    image_height: int | None = None
    expected_labels: list[str] = []
    missing_labels: list[str] = []
    completeness: float | None = None  # found expected labels / expected labels
    mean_confidence: float | None = None
    warnings: list[str] = []

    def regions_for(self, label: str) -> list[DetectedRegion]:
        return sorted(
            (r for r in self.regions if r.label == label),
            key=lambda r: r.bbox.confidence,
            reverse=True,
        )

    def best_region(self, label: str) -> DetectedRegion | None:
        matches = self.regions_for(label)
        return matches[0] if matches else None


class RegionTamperingFinding(BaseModel):
    label: str
    ela_hot_ratio: float
    ela_mean_error: float
    copy_move_hits: int
    suspicious: bool
    reason: str | None = None


class TamperingSummary(BaseModel):
    tampering_score: int
    verdict: Literal["clean", "suspicious", "tampered"]
    evidence: list[str]
    ela_suspicious: bool
    ela_max_error: float
    copy_move_matches: int
    ela_heatmap: str | None = None  # data: URL, present only if generated
    region_findings: list[RegionTamperingFinding] = []


class FaceSummary(BaseModel):
    attempted: bool
    similarity: float | None = None
    is_match: bool | None = None
    backend: str | None = None
    document_face_found: bool = False
    live_face_found: bool = False
    document_face_source: str | None = None  # "photo_region" | "full_image"


class RiskSummary(BaseModel):
    risk_score: int
    verdict: VerdictLiteral
    contributing_factors: list[str]
    validation_score: int
    tampering_score: int
    face_similarity: float | None = None
    localization_completeness: float | None = None


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
    # Optional so scans stored before localization existed still deserialize.
    localization: RegionLocalizationResult | None = None


class ScanListItem(BaseModel):
    id: str
    timestamp: str
    document_type: DocumentTypeLiteral
    risk_score: int
    verdict: VerdictLiteral


class StatsResponse(BaseModel):
    total_scans: int
    by_verdict: dict[str, int]
