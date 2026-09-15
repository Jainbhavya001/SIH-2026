import json
import logging
from pathlib import Path
from dataclasses import asdict

from app.utils.image_utils import bytes_to_bgr
from app.modules.ocr.engine import run_ocr
from app.modules.ocr.mrz_parser import parse_mrz
from app.modules.ocr.field_extractors import extract_fields
from app.modules.validation.rules_engine import validate_passport
from app.modules.tampering.tampering_engine import analyze_tampering
from app.modules.face.detector import detect_largest_face
from app.modules.face.verifier import compare_faces
from app.modules.risk.risk_engine import compute_risk

logging.basicConfig(level=logging.INFO)

image_path = Path(r"d:\Projects\SIH26\SIH-2026\WhatsApp Image 2026-09-12 at 11.28.17 PM.jpeg")
image_bytes = image_path.read_bytes()
bgr_img = bytes_to_bgr(image_bytes)

print("=== 1. OCR RUN ===")
ocr_res = run_ocr(bgr_img)
print(f"Backend: {ocr_res.backend}")
print(f"Engine Available: {ocr_res.engine_available}")
print(f"Mean Confidence: {ocr_res.mean_confidence}")
print(f"Raw Text:\n{ocr_res.raw_text}\n")

print("=== 2. MRZ PARSER ===")
mrz_res = parse_mrz(ocr_res.raw_text)
print(f"MRZ Detected: {mrz_res.detected}")
print(f"Format: {mrz_res.format}")
print(f"Issuing Country: {mrz_res.issuing_country}")
print(f"Surname: {mrz_res.surname}")
print(f"Given Names: {mrz_res.given_names}")
print(f"Nationality: {mrz_res.nationality}")
print(f"DOB: {mrz_res.date_of_birth}")
print(f"Expiry: {mrz_res.date_of_expiry}")
print(f"Composite Valid: {mrz_res.composite_valid}")
print("Fields:")
for f in mrz_res.fields:
    print(f"  - {f.name}: value='{f.value}', valid={f.valid}")
print(f"Warnings: {mrz_res.warnings}")

print("\n=== 3. FIELD EXTRACTORS ===")
extracted = extract_fields("passport", ocr_res.raw_text)
print(f"Extracted visual fields: {extracted}")

print("\n=== 4. VALIDATION ENGINE ===")
visual_name = extracted.get("name")
val_res = validate_passport(mrz_res, visual_name=visual_name)
print(f"Validation Score: {val_res.score}")
print(f"Critical Count: {val_res.critical_count}")
print("Issues:")
for issue in val_res.issues:
    print(f"  - [{issue.severity}] {issue.code}: {issue.message} (field={issue.field})")

print("\n=== 5. TAMPERING ENGINE ===")
tamp_res = analyze_tampering(image_bytes, bgr_img)
print(f"Tampering Score: {tamp_res.tampering_score}")
print(f"Verdict: {tamp_res.verdict}")
print(f"Evidence: {tamp_res.evidence}")
if tamp_res.ela:
    print(f"ELA suspicious: {tamp_res.ela.suspicious}, max_error={tamp_res.ela.max_error:.2f}, mean_error={tamp_res.ela.mean_error:.2f}")
if tamp_res.copy_move:
    print(f"Copy-move matches: {tamp_res.copy_move.match_count}, suspicious={tamp_res.copy_move.suspicious}")

print("\n=== 6. FACE DETECTION ===")
faces = detect_largest_face(bgr_img)
print(f"Detected face count: {len(faces) if faces else 0}")

print("\n=== 7. RISK ENGINE ===")
risk_res = compute_risk(val_res, tamp_res, None)
print(f"Risk Score: {risk_res.risk_score}")
print(f"Verdict: {risk_res.verdict}")
print(f"Flags: {risk_res.flags}")
