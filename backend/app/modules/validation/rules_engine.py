"""
Module 2 — Document Validation.

Runs a battery of explainable rules over the OCR/MRZ output and reports each
as a `ValidationIssue`. We deliberately never produce a single opaque
pass/fail — every finding carries a human-readable reason, because an
officer (or an auditor reviewing the digital trail later) needs to know
*why* a document was flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher

from app.modules.ocr.mrz_parser import MRZResult
from app.modules.validation.schemas import KNOWN_COUNTRY_CODES

Severity = str  # "info" | "warning" | "critical"

_SEVERITY_PENALTY = {"info": 2, "warning": 10, "critical": 30}


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: Severity
    field: str | None = None


@dataclass
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)
    score: int = 100  # 100 = fully valid, 0 = fails everything checked
    checks_run: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


def _parse_iso(d: str | None) -> date | None:
    if not d:
        return None
    try:
        return datetime.fromisoformat(d).date()
    except ValueError:
        return None


def _name_similarity(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 1.0  # nothing to compare, don't penalize
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _char_differences(a: str, b: str) -> int:
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y)
    return max(len(a), len(b)) - int(round(SequenceMatcher(None, a, b).ratio() * max(len(a), len(b))))


def _compare_visual_to_mrz(
    field_name: str, label: str, mrz_value: str | None, visual_value: str | None
) -> ValidationIssue | None:
    """
    Visual-zone vs MRZ consistency. A forger who edits the printed data page
    rarely regenerates a checksum-valid MRZ to match, so a disagreement is a
    classic alteration signal. A single-character difference is reported as
    a warning because it is also what one OCR misread looks like.
    """
    if not mrz_value or not visual_value:
        return None
    a = mrz_value.replace("<", "").replace("-", "").upper()
    b = visual_value.replace("<", "").replace("-", "").upper()
    diff = _char_differences(a, b)
    if diff == 0:
        return None
    if diff == 1:
        return ValidationIssue(
            code=f"{field_name.upper()}_MINOR_MISMATCH_MRZ_VS_VISUAL",
            message=f"{label} in the MRZ ('{mrz_value}') differs by one character from the printed value ('{visual_value}') — likely an OCR misread, verify manually.",
            severity="warning",
            field=field_name,
        )
    return ValidationIssue(
        code=f"{field_name.upper()}_MISMATCH_MRZ_VS_VISUAL",
        message=f"{label} in the MRZ ('{mrz_value}') does not match the printed value ('{visual_value}'). Possible alteration of the data page.",
        severity="critical",
        field=field_name,
    )


def validate_passport(
    mrz: MRZResult,
    visual_name: str | None = None,
    visual_fields: dict | None = None,
) -> ValidationResult:
    """
    `visual_fields` should contain only values read from localized regions
    (Module 0); full-page positional guesses are too unreliable to cross-check
    against the MRZ.
    """
    issues: list[ValidationIssue] = []
    checks = 0

    if not mrz.detected:
        issues.append(ValidationIssue(
            code="MRZ_NOT_FOUND",
            message="No machine-readable zone detected — cannot cryptographically verify document number, date of birth, or expiry.",
            severity="critical",
        ))
        return _finalize(issues, checks_run=1)

    for f in mrz.fields:
        checks += 1
        if f.valid is False:
            issues.append(ValidationIssue(
                code=f"MRZ_CHECKSUM_{f.name.upper()}",
                message=f"MRZ check digit for '{f.name}' does not match (expected {f.check_digit_expected}, computed {f.check_digit_found}). This field may have been altered.",
                severity="critical",
                field=f.name,
            ))
        elif f.valid is None:
            issues.append(ValidationIssue(
                code=f"MRZ_CHECKSUM_UNREADABLE_{f.name.upper()}",
                message=f"Could not compute a check digit for '{f.name}' — OCR text may be too noisy.",
                severity="warning",
                field=f.name,
            ))

    checks += 1
    if mrz.composite_valid is False:
        issues.append(ValidationIssue(
            code="MRZ_COMPOSITE_CHECKSUM",
            message="MRZ composite (overall) check digit failed. Strong indicator of tampering or a fraudulent document.",
            severity="critical",
        ))

    checks += 1
    expiry = _parse_iso(mrz.date_of_expiry)
    if expiry and expiry < date.today():
        issues.append(ValidationIssue(
            code="DOCUMENT_EXPIRED",
            message=f"Passport expired on {expiry.isoformat()}.",
            severity="critical",
            field="date_of_expiry",
        ))
    elif not expiry:
        issues.append(ValidationIssue(
            code="EXPIRY_UNREADABLE",
            message="Could not parse a valid expiry date from the MRZ.",
            severity="warning",
            field="date_of_expiry",
        ))

    checks += 1
    dob = _parse_iso(mrz.date_of_birth)
    if dob:
        age_days = (date.today() - dob).days
        age_years = age_days / 365.25
        if age_years < 0 or age_years > 120:
            issues.append(ValidationIssue(
                code="IMPLAUSIBLE_AGE",
                message=f"Date of birth implies an implausible age ({age_years:.0f} years).",
                severity="critical",
                field="date_of_birth",
            ))
    else:
        issues.append(ValidationIssue(
            code="DOB_UNREADABLE",
            message="Could not parse a valid date of birth from the MRZ.",
            severity="warning",
            field="date_of_birth",
        ))

    checks += 1
    if mrz.nationality and mrz.nationality not in KNOWN_COUNTRY_CODES:
        issues.append(ValidationIssue(
            code="UNRECOGNIZED_NATIONALITY_CODE",
            message=f"Nationality code '{mrz.nationality}' is not in the recognized reference set (prototype uses a partial ISO 3166-1 alpha-3 list — verify manually).",
            severity="info",
            field="nationality",
        ))

    checks += 1
    mrz_full_name = " ".join(filter(None, [mrz.given_names, mrz.surname]))
    similarity = _name_similarity(mrz_full_name, visual_name)
    if similarity < 0.55:
        issues.append(ValidationIssue(
            code="NAME_MISMATCH_MRZ_VS_VISUAL",
            message=f"Name in the MRZ ('{mrz_full_name}') does not closely match the printed name elsewhere on the document ('{visual_name}'). Possible substitution.",
            severity="critical",
            field="name",
        ))

    if visual_fields:
        mrz_passport_number = next(
            (f.value.replace("<", "") for f in mrz.fields if f.name == "passport_number"), None
        )
        comparisons = [
            ("passport_number", "Passport number", mrz_passport_number, visual_fields.get("passport_number")),
            ("date_of_birth", "Date of birth", mrz.date_of_birth, visual_fields.get("date_of_birth")),
            ("date_of_expiry", "Date of expiry", mrz.date_of_expiry, visual_fields.get("date_of_expiry")),
        ]
        for field_name, label, mrz_value, visual_value in comparisons:
            if not mrz_value or not visual_value:
                continue
            checks += 1
            issue = _compare_visual_to_mrz(field_name, label, mrz_value, visual_value)
            if issue:
                issues.append(issue)

    return _finalize(issues, checks_run=checks)


def validate_generic(document_type: str, extracted_fields: dict) -> ValidationResult:
    """Lighter-weight rule pass for documents without an MRZ (visa/ID/DL/permit)."""
    issues: list[ValidationIssue] = []
    checks = 0

    required_by_type = {
        "visa": ["visa_number", "visa_type"],
        "national_id": ["name", "id_number"],
        "driving_license": ["name", "license_number"],
        "permit": ["permit_number"],
    }
    for required_field in required_by_type.get(document_type, []):
        checks += 1
        if not extracted_fields.get(required_field):
            issues.append(ValidationIssue(
                code=f"MISSING_FIELD_{required_field.upper()}",
                message=f"Could not extract required field '{required_field}' from the document.",
                severity="warning",
                field=required_field,
            ))

    # Expiry-in-the-past check, wherever an expiry-like field exists.
    for key in ("expiry_date", "valid_till"):
        if key in extracted_fields:
            checks += 1
            d = _parse_iso(extracted_fields.get(key))
            if d and d < date.today():
                issues.append(ValidationIssue(
                    code="DOCUMENT_EXPIRED",
                    message=f"Document expired on {d.isoformat()}.",
                    severity="critical",
                    field=key,
                ))

    # Issue date after expiry date is a logical impossibility.
    if "issue_date" in extracted_fields and "expiry_date" in extracted_fields:
        checks += 1
        issue_d = _parse_iso(extracted_fields.get("issue_date"))
        expiry_d = _parse_iso(extracted_fields.get("expiry_date"))
        if issue_d and expiry_d and issue_d > expiry_d:
            issues.append(ValidationIssue(
                code="ISSUE_AFTER_EXPIRY",
                message="Issue date is after the expiry date — internally inconsistent document.",
                severity="critical",
            ))

    return _finalize(issues, checks_run=max(checks, 1))


def _finalize(issues: list[ValidationIssue], checks_run: int) -> ValidationResult:
    penalty = sum(_SEVERITY_PENALTY[i.severity] for i in issues)
    score = max(0, 100 - penalty)
    return ValidationResult(issues=issues, score=score, checks_run=checks_run)
