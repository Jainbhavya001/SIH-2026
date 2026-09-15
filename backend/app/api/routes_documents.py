"""
The main pipeline endpoint: upload a document (+ optional live selfie) and
get back a full localization -> OCR -> validation -> tampering -> face ->
risk report.

This route is deliberately a thin orchestrator — it calls each module in
turn and assembles the response. All the actual logic lives in
`app/modules/*`, which keeps this file readable as a map of the pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.models.schemas import (
    FaceSummary,
    MRZSummary,
    OCRSummary,
    RegionLocalizationResult,
    RegionTamperingFinding,
    RiskSummary,
    ScanListItem,
    ScanResponse,
    StatsResponse,
    TamperingSummary,
    ValidationSummary,
)
from app.modules.face.detector import detect_largest_face
from app.modules.face.verifier import compare_faces
from app.modules.localization.config import TEXT_FIELD_LABELS
from app.modules.localization.detector import detect_document_regions, trusted_regions
from app.modules.ocr.engine import run_ocr, run_region_ocr
from app.modules.ocr.field_extractors import extract_fields, merge_region_fields, region_field_values
from app.modules.ocr.mrz_parser import mrz_quality, parse_mrz
from app.modules.risk.risk_engine import compute_risk
from app.modules.tampering.tampering_engine import analyze_tampering
from app.modules.validation.rules_engine import validate_generic, validate_passport
from app.storage import file_store
from app.utils.image_utils import bytes_to_bgr, png_bytes_to_data_url

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

_MAX_BYTES_DEFAULT = 12 * 1024 * 1024


async def _read_upload(upload: UploadFile) -> bytes:
    settings = get_settings()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    data = await upload.read()
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_MB}MB limit.")
    if not data:
        raise HTTPException(400, "Empty file upload.")
    return data


def _localize(doc_image: np.ndarray, document_type: str) -> RegionLocalizationResult | None:
    """Module 0 must never fail a scan: an unexpected error degrades to the
    pre-localization pipeline (full-image processing everywhere)."""
    try:
        return detect_document_regions(doc_image, document_type=document_type)
    except Exception:
        logger.exception("Region localization failed; continuing with full-image processing.")
        return None


@router.post("/scan", response_model=ScanResponse)
async def scan_document(
    document_type: str = Form(...),
    document: UploadFile = File(...),
    live_face: UploadFile | None = File(None),
):
    valid_types = {"passport", "visa", "national_id", "driving_license", "permit", "pan_card", "aadhaar"}
    if document_type not in valid_types:
        raise HTTPException(400, f"document_type must be one of {sorted(valid_types)}")

    doc_bytes = await _read_upload(document)
    try:
        doc_image = bytes_to_bgr(doc_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    # --- Module 0: Region localization (Faster R-CNN, CPU fallbacks) ---
    # Contrast normalisation happens inside the detector on a copy, so the
    # forensics below still see original pixels and boxes stay aligned.
    localization = _localize(doc_image, document_type)
    regions = trusted_regions(localization)
    photo_region = localization.best_region("photo") if localization else None
    if photo_region is not None and photo_region not in regions:
        photo_region = None

    # --- Module 1: OCR (full page for label context + localized crops) ---
    ocr_result = run_ocr(doc_image)
    ocr_labels = set(TEXT_FIELD_LABELS)
    if document_type in ("passport", "visa"):
        ocr_labels.add("mrz")
    region_ocr = (
        run_region_ocr(doc_image, regions, labels=ocr_labels)
        if regions and ocr_result.engine_available
        else {}
    )
    region_text = {label: r.text for label, r in region_ocr.items()}
    field_text = {label: text for label, text in region_text.items() if label != "mrz"}

    extracted_fields = merge_region_fields(
        document_type, extract_fields(document_type, ocr_result.raw_text), field_text
    )
    mrz_summary: MRZSummary | None = None
    mrz_full = None

    if document_type in ("passport", "visa"):
        mrz_full = parse_mrz(ocr_result.raw_text)
        mrz_source = "full_image"
        if "mrz" in region_text:
            region_mrz = parse_mrz(region_text["mrz"])
            if mrz_quality(region_mrz) > mrz_quality(mrz_full):
                mrz_full, mrz_source = region_mrz, "mrz_region"
        mrz = mrz_full
        mrz_summary = MRZSummary(
            detected=mrz.detected,
            format=mrz.format,
            issuing_country=mrz.issuing_country,
            surname=mrz.surname,
            given_names=mrz.given_names,
            nationality=mrz.nationality,
            sex=mrz.sex,
            date_of_birth=mrz.date_of_birth,
            date_of_expiry=mrz.date_of_expiry,
            composite_valid=mrz.composite_valid,
            fields=[{"name": f.name, "value": f.value, "valid": f.valid} for f in mrz.fields],
            warnings=mrz.warnings,
            source=mrz_source if mrz.detected else None,
        )

    # --- Module 2: Validation ---
    if document_type == "passport" and mrz_full is not None:
        visual_name = extracted_fields.get("name")
        validation = validate_passport(
            mrz_full,
            visual_name=visual_name,
            visual_fields=region_field_values(document_type, field_text) or None,
        )
    else:
        validation = validate_generic(document_type, extracted_fields)

    # --- Module 3: Tampering (whole image + localized high-risk regions) ---
    tampering = analyze_tampering(doc_bytes, doc_image, regions=regions)

    # --- Module 4: Face verification (optional — needs a live capture) ---
    face_summary = FaceSummary(attempted=False)
    face_match = None
    doc_face_crop = detect_largest_face(
        doc_image, photo_region=photo_region.bbox if photo_region else None
    )
    doc_face_source = doc_face_crop[1].source if doc_face_crop else None

    if live_face is not None:
        live_bytes = await _read_upload(live_face)
        try:
            live_image = bytes_to_bgr(live_bytes)
        except ValueError as e:
            raise HTTPException(400, f"Live capture: {e}") from e
        live_face_crop = detect_largest_face(live_image)

        face_match = compare_faces(
            doc_face_crop[0] if doc_face_crop else None,
            live_face_crop[0] if live_face_crop else None,
        )
        face_summary = FaceSummary(
            attempted=True,
            similarity=face_match.similarity,
            is_match=face_match.is_match,
            backend=face_match.backend,
            document_face_found=face_match.document_face_found,
            live_face_found=face_match.live_face_found,
            document_face_source=doc_face_source,
        )
    else:
        face_summary.document_face_found = doc_face_crop is not None
        face_summary.document_face_source = doc_face_source

    # --- Risk Engine ---
    risk = compute_risk(
        validation, tampering, face_match, localization=localization, document_type=document_type
    )

    ela_heatmap_url = None
    if tampering.ela and tampering.ela.heatmap_png:
        ela_heatmap_url = png_bytes_to_data_url(tampering.ela.heatmap_png)

    response = ScanResponse(
        id="pending",
        timestamp="pending",
        document_type=document_type,
        ocr=OCRSummary(
            raw_text=ocr_result.raw_text,
            mean_confidence=ocr_result.mean_confidence,
            engine_available=ocr_result.engine_available,
            warning=ocr_result.warning,
            extracted_fields=extracted_fields,
            region_text=region_text,
        ),
        mrz=mrz_summary,
        validation=ValidationSummary(
            score=validation.score,
            issues=[
                {"code": i.code, "message": i.message, "severity": i.severity, "field": i.field}
                for i in validation.issues
            ],
            checks_run=validation.checks_run,
        ),
        tampering=TamperingSummary(
            tampering_score=tampering.tampering_score,
            verdict=tampering.verdict,
            evidence=tampering.evidence,
            ela_suspicious=tampering.ela.suspicious if tampering.ela else False,
            ela_max_error=tampering.ela.max_error if tampering.ela else 0.0,
            copy_move_matches=tampering.copy_move.match_count if tampering.copy_move else 0,
            ela_heatmap=ela_heatmap_url,
            region_findings=[
                RegionTamperingFinding(
                    label=f.label,
                    ela_hot_ratio=f.ela_hot_ratio,
                    ela_mean_error=f.ela_mean_error,
                    copy_move_hits=f.copy_move_hits,
                    suspicious=f.suspicious,
                    reason=f.reason,
                )
                for f in tampering.region_findings
            ],
        ),
        face=face_summary,
        risk=RiskSummary(**asdict(risk)),
        localization=localization,
    )

    record = response.model_dump()
    scan_id = file_store.save_scan(record)
    record = file_store.get_scan(scan_id)
    return ScanResponse(**record)


@router.get("/scans", response_model=list[ScanListItem])
async def list_scans(limit: int = 50):
    records = file_store.list_scans(limit=limit)
    return [
        ScanListItem(
            id=r["id"],
            timestamp=r["timestamp"],
            document_type=r["document_type"],
            risk_score=r["risk"]["risk_score"],
            verdict=r["risk"]["verdict"],
        )
        for r in records
    ]


@router.get("/scans/{scan_id}", response_model=ScanResponse)
async def get_scan(scan_id: str):
    record = file_store.get_scan(scan_id)
    if not record:
        raise HTTPException(404, "Scan not found.")
    return ScanResponse(**record)


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    return StatsResponse(**file_store.stats())
