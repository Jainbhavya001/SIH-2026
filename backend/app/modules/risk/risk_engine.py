"""
Risk Engine — combines Module 2 (validation), Module 3 (tampering), and
Module 4 (face verification) outputs into one transparent 0-100 risk score
and a three-tier verdict an officer can act on immediately.

Explainability is a hard requirement here: this is a decision-support tool
for a security officer, not an autonomous gate. Every point of risk must be
traceable to a specific piece of evidence in `contributing_factors`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings
from app.modules.face.verifier import FaceMatchResult
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


def compute_risk(
    validation: ValidationResult,
    tampering: TamperingResult,
    face_match: FaceMatchResult | None = None,
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

    weighted = (
        validation_risk * settings.WEIGHT_VALIDATION
        + tampering_risk * settings.WEIGHT_TAMPERING
        + face_risk * settings.WEIGHT_FACE
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
    )
