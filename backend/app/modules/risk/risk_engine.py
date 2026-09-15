"""
Risk Engine — combines Module 2 (validation), Module 3 (tampering), and
Module 4 (face verification) outputs into one transparent 0-100 risk score
and a three-tier verdict an officer can act on immediately.

Module 0 (region localization) adds a small, capped adjustment when a
trained Faster R-CNN model reports that regions a genuine document of this
type always carries (e.g. the portrait photo or the MRZ) could not be found.
Heuristic fallback boxes never add risk: their absence says more about the
fallback than about the document.

Explainability is a hard requirement here: this is a decision-support tool
for a security officer, not an autonomous gate. Every point of risk must be
traceable to a specific piece of evidence in `contributing_factors`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.models.schemas import RegionLocalizationResult
from app.modules.face.verifier import FaceMatchResult
from app.modules.localization.config import CRITICAL_REGIONS, FALLBACK_LABELS
from app.modules.tampering.tampering_engine import TamperingResult
from app.modules.validation.rules_engine import ValidationResult


@dataclass
class RiskReport:
    risk_score: int  # 0 (low risk) - 100 (high risk)
    verdict: str  # "CLEAR" | "REVIEW" | "REJECT"
    contributing_factors: list[str] = field(default_factory=list)
    validation_score: int = 100
    tampering_score: int = 0
    face_similarity: float | None = None
    localization_completeness: float | None = None


def _localization_risk(
    localization: RegionLocalizationResult | None,
    document_type: str | None,
    settings: Settings,
    factors: list[str],
) -> float:
    if localization is None or localization.backend != "faster_rcnn" or localization.completeness is None:
        return 0.0

    points = (1.0 - localization.completeness) * settings.LOCALIZATION_RISK_MAX_POINTS
    missing_critical = [
        label for label in CRITICAL_REGIONS.get(document_type or "", []) if label in localization.missing_labels
    ]
    if missing_critical:
        points += settings.LOCALIZATION_MISSING_CRITICAL_POINTS * len(missing_critical)
        factors.append(
            f"Region localization could not find the {', '.join(missing_critical)} region(s) expected on a "
            f"{(document_type or 'document').replace('_', ' ')} — layout does not match a genuine document."
        )
    elif localization.completeness < 0.7:
        factors.append(
            f"Only {localization.completeness * 100:.0f}% of the regions expected on this document type were "
            f"localized (missing: {', '.join(localization.missing_labels)})."
        )

    weak = [
        r.label for r in localization.regions
        if r.label not in FALLBACK_LABELS and r.bbox.confidence < settings.LOCALIZATION_LOW_CONFIDENCE
    ]
    if len(weak) >= 3:
        points += 3
        factors.append(
            f"{len(weak)} localized regions have low detection confidence ({', '.join(sorted(set(weak)))}) — "
            f"print quality or layout deviates from reference documents."
        )

    return min(float(settings.LOCALIZATION_RISK_MAX_POINTS), points)


def compute_risk(
    validation: ValidationResult,
    tampering: TamperingResult,
    face_match: FaceMatchResult | None = None,
    localization: RegionLocalizationResult | None = None,
    document_type: str | None = None,
) -> RiskReport:
    settings = get_settings()
    factors: list[str] = []

    # --- Validation contribution ---
    validation_risk = 100 - validation.score
    if validation.critical_count:
        factors.append(
            f"{validation.critical_count} critical validation failure(s): "
            + "; ".join(i.message for i in validation.issues if i.severity == "critical")
        )
    if validation.warning_count:
        factors.append(f"{validation.warning_count} field(s) flagged for manual review.")

    # --- Tampering contribution ---
    tampering_risk = tampering.tampering_score
    if tampering.verdict != "clean":
        factors.extend(tampering.evidence)

    # --- Face contribution ---
    if face_match is not None and (face_match.document_face_found or face_match.live_face_found):
        if not face_match.document_face_found:
            face_risk = 40.0
            factors.append("No face detected on the document photo — cannot verify identity.")
        elif not face_match.live_face_found:
            face_risk = 40.0
            factors.append("No live capture provided/detected — face verification skipped.")
        elif not face_match.is_match:
            face_risk = round((1 - face_match.similarity) * 100)
            factors.append(
                f"Face similarity {face_match.similarity * 100:.0f}% is below the match threshold "
                f"({settings.FACE_MATCH_THRESHOLD * 100:.0f}%) — presented individual may not match the document."
            )
        else:
            face_risk = round((1 - face_match.similarity) * 40)  # matched, but scale small residual risk
    else:
        face_risk = 0.0  # no live capture attempted for this scan — not itself a red flag

    # --- Localization contribution (additive and capped) ---
    localization_risk = _localization_risk(localization, document_type, settings, factors)

    weighted = (
        validation_risk * settings.WEIGHT_VALIDATION
        + tampering_risk * settings.WEIGHT_TAMPERING
        + face_risk * settings.WEIGHT_FACE
        + localization_risk
    )
    risk_score = int(round(min(100, max(0, weighted))))

    # A hard override: any critical validation failure or a "tampered" verdict
    # cannot be washed out by a good face match — these are disqualifying on
    # their own for a border-security context.
    if validation.critical_count > 0 or tampering.verdict == "tampered":
        risk_score = max(risk_score, 66)

    if risk_score <= settings.RISK_CLEAR_MAX:
        verdict = "CLEAR"
    elif risk_score <= settings.RISK_REVIEW_MAX:
        verdict = "REVIEW"
    else:
        verdict = "REJECT"

    if not factors:
        factors.append("No risk indicators found across validation, tampering, or face-match checks.")

    return RiskReport(
        risk_score=risk_score,
        verdict=verdict,
        contributing_factors=factors,
        validation_score=validation.score,
        tampering_score=tampering.tampering_score,
        face_similarity=face_match.similarity if face_match else None,
        localization_completeness=localization.completeness if localization else None,
    )
