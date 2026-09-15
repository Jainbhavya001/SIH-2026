"""
Machine Readable Zone (MRZ) parsing per ICAO Doc 9303.

This is the one piece of the pipeline that is fully deterministic and
standards-based rather than heuristic: passports (TD3 format) print a
two-line, 44-characters-per-line MRZ at the bottom of the photo page, and
each numeric field is protected by a check digit computed with a published
weighting algorithm. If OCR reads the MRZ correctly, we can *mathematically
prove* whether the printed number, date of birth, and expiry date are
internally consistent — this is exactly the kind of signal a border officer
cannot get from squinting at a passport.

Reference: ICAO Doc 9303, Part 4, Section 4.9 (check digit algorithm) and
Section 4.2.2 (TD3 MRZ layout for passports).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_WEIGHTS = (7, 3, 1)


def _char_value(ch: str) -> int:
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    if ch.isalpha():
        return ord(ch.upper()) - ord("A") + 10
    raise ValueError(f"Invalid MRZ character: {ch!r}")


def compute_check_digit(data: str) -> int:
    """ICAO 9303 check-digit algorithm: weighted sum mod 10, weights 7-3-1 repeating."""
    total = 0
    for i, ch in enumerate(data):
        total += _char_value(ch) * _WEIGHTS[i % 3]
    return total % 10


def _clean_line(line: str) -> str:
    # OCR frequently confuses '<' with 'K', 'C', or drops it; normalize common
    # noise but keep this conservative so we don't hide genuine tampering.
    return re.sub(r"[^A-Z0-9<]", "", line.upper())


@dataclass
class MRZField:
    name: str
    value: str
    check_digit_expected: int | None = None
    check_digit_found: int | None = None

    @property
    def valid(self) -> bool | None:
        if self.check_digit_expected is None:
            return None
        return self.check_digit_expected == self.check_digit_found


@dataclass
class MRZResult:
    detected: bool
    format: str | None = None  # "TD3" (passport/visa) etc.
    document_type: str | None = None
    issuing_country: str | None = None
    surname: str | None = None
    given_names: str | None = None
    fields: list[MRZField] = field(default_factory=list)
    composite_valid: bool | None = None
    raw_lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    nationality: str | None = None
    sex: str | None = None
    date_of_birth: str | None = None  # ISO yyyy-mm-dd, best-effort century guess
    date_of_expiry: str | None = None

    @property
    def all_checks_pass(self) -> bool:
        if not self.detected:
            return False
        field_checks = [f.valid for f in self.fields if f.valid is not None]
        checks = field_checks + ([self.composite_valid] if self.composite_valid is not None else [])
        return bool(checks) and all(checks)


def _clean_mrz_line1(raw: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9<]", "", raw.upper())
    # Fix <K< or <C< misread for << separator
    cleaned = re.sub(r"<[KC]<", "<<", cleaned)
    if "<<" in cleaned:
        prefix, _, given_and_fillers = cleaned.partition("<<")
        m = re.match(r"^([A-Z<]+?)(?:[<K]{3,}|$)", given_and_fillers)
        if m:
            given = m.group(1).replace("K", "<")
            cleaned = prefix + "<<" + given
    return (cleaned + "<" * 44)[:44]


def _clean_mrz_line2(raw: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9<]", "", raw.upper())
    # Fix common 1ND -> IND for country code at pos 10:13
    if len(cleaned) >= 13 and cleaned[10:13] == "1ND":
        cleaned = cleaned[:10] + "IND" + cleaned[13:]

    # Normalize common OCR letter-to-digit misreads in strictly numeric positions of Line 2:
    # DOB: pos 13..19 (6 chars) + check digit (pos 19)
    # Expiry: pos 21..27 (6 chars) + check digit (pos 27)
    # Check digits at pos 9, 42, 43
    char_list = list((cleaned + "<" * 44)[:44])
    digit_positions = list(range(13, 20)) + list(range(21, 28)) + [9, 42, 43]
    char_map = {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}

    for pos in digit_positions:
        if pos < len(char_list) and char_list[pos] in char_map:
            char_list[pos] = char_map[char_list[pos]]

    return "".join(char_list)


def _is_valid_td3_line1_candidate(line: str) -> bool:
    clean = re.sub(r"[^A-Z0-9<]", "", line.upper())
    if len(clean) < 30 or len(clean) > 55:
        return False
    if not clean.startswith("P"):
        return False
    # MRZ Line 1 must contain '<<' separator or at least 3 '<' filler characters
    if "<<" not in clean and clean.count("<") < 3:
        return False
    return True


def _is_valid_td3_line2_candidate(line: str) -> bool:
    clean = re.sub(r"[^A-Z0-9<]", "", line.upper())
    if len(clean) < 30 or len(clean) > 55:
        return False
    if clean.startswith("P<"):
        return False
    padded = (clean + "<" * 44)[:44]
    # Sex character at index 20 must be M, F, X, or <
    if padded[20] not in ("M", "F", "X", "<"):
        return False
    digit_like_count = sum(1 for c in clean if c.isdigit() or c in "OIZSBQ")
    if digit_like_count < 8:
        return False
    return True


def _find_td3_lines(raw_text: str) -> list[str] | None:
    """
    Look for two 44-character MRZ lines inside OCR output.
    Enforces strict structural checks to avoid matching visual-zone text.
    """
    raw_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    line1_candidate = None
    line2_candidate = None

    # 1. Primary: search line pairs in raw_lines
    for i, line in enumerate(raw_lines):
        if _is_valid_td3_line1_candidate(line):
            line1_cand = line
            for j in range(i + 1, min(i + 4, len(raw_lines))):
                if _is_valid_td3_line2_candidate(raw_lines[j]):
                    line1_candidate = line1_cand
                    line2_candidate = raw_lines[j]
                    break
            if line2_candidate:
                break

    # 2. Fallback: regex search across raw_text if line-by-line check fails
    if not line1_candidate or not line2_candidate:
        m = re.search(
            r"(P<[A-Z0-9<K]{2,}\s*<<[A-Z0-9<K\s]{15,60})\s+([A-Z0-9]{7,10}[<A-Z0-9\s]{25,60})",
            raw_text,
        )
        if m:
            l1_raw, l2_raw = m.group(1), m.group(2)
            if _is_valid_td3_line1_candidate(l1_raw) and _is_valid_td3_line2_candidate(l2_raw):
                line1_candidate, line2_candidate = l1_raw, l2_raw

    if not line1_candidate or not line2_candidate:
        return None

    line1 = _clean_mrz_line1(line1_candidate)
    line2 = _clean_mrz_line2(line2_candidate)

    # Final structural verification of cleaned 44-character strings
    if not line1.startswith("P") or "<" not in line1:
        return None
    if sum(1 for c in line2 if c.isdigit()) < 6:
        return None

    return [line1, line2]


def parse_mrz(raw_text: str) -> MRZResult:
    lines = _find_td3_lines(raw_text)
    if not lines:
        return MRZResult(detected=False, warnings=["No MRZ block detected in OCR output."])

    line1, line2 = lines
    warnings: list[str] = []

    if not line1.startswith(("P<", "P")):
        warnings.append("Line 1 does not start with the expected document code 'P'.")

    doc_type = line1[0:2].replace("<", "")
    issuing_country = line1[2:5].replace("<", "")

    name_field = line1[5:44]
    surname, _, given = name_field.partition("<<")
    surname = surname.replace("<", " ").strip()
    given_names = given.replace("<", " ").strip()

    passport_number_raw = line2[0:9]
    passport_check = line2[9]
    nationality = line2[10:13].replace("<", "")
    dob_raw = line2[13:19]
    dob_check = line2[19]
    sex = line2[20]
    expiry_raw = line2[21:27]
    expiry_check = line2[27]
    personal_number_raw = line2[28:42]
    personal_check = line2[42]

    def make_field(name: str, value: str, expected_digit_char: str) -> MRZField:
        try:
            expected = int(expected_digit_char)
        except ValueError:
            expected = None
        try:
            found = compute_check_digit(value)
        except ValueError:
            found = None
        return MRZField(name=name, value=value, check_digit_expected=expected, check_digit_found=found)

    fields = [
        make_field("passport_number", passport_number_raw, passport_check),
        make_field("date_of_birth", dob_raw, dob_check),
        make_field("date_of_expiry", expiry_raw, expiry_check),
    ]
    if personal_number_raw.strip("<"):
        fields.append(make_field("personal_number", personal_number_raw, personal_check))

    # Composite check digit covers positions 0-9, 13-19, 21-42 of line 2.
    composite_input = line2[0:10] + line2[13:20] + line2[21:43]
    composite_check_char = line2[43] if len(line2) > 43 else "0"
    composite_valid: bool | None
    try:
        composite_expected = int(composite_check_char)
        composite_valid = compute_check_digit(composite_input) == composite_expected
    except ValueError:
        composite_valid = None

    def fmt_date(raw: str, is_expiry: bool = False) -> str | None:
        if len(raw) != 6 or not raw.isdigit():
            return None
        yy, mm, dd = raw[0:2], raw[2:4], raw[4:6]
        if is_expiry:
            century = "20"
        else:
            century = "19" if int(yy) > 26 else "20"
        return f"{century}{yy}-{mm}-{dd}"

    return MRZResult(
        detected=True,
        format="TD3",
        document_type=doc_type or "P",
        issuing_country=issuing_country,
        surname=surname or None,
        given_names=given_names or None,
        fields=fields,
        composite_valid=composite_valid,
        raw_lines=[line1, line2],
        warnings=warnings,
        nationality=nationality or None,
        sex=sex if sex in ("M", "F") else None,
        date_of_birth=fmt_date(dob_raw, is_expiry=False),
        date_of_expiry=fmt_date(expiry_raw, is_expiry=True),
    )


def mrz_quality(result: MRZResult) -> tuple[int, int, int]:
    """Sort key for competing MRZ reads (e.g. localized crop vs full page):
    detected first, then composite check passing, then valid check digits."""
    return (
        int(result.detected),
        int(result.composite_valid is True),
        sum(1 for f in result.fields if f.valid is True),
    )
