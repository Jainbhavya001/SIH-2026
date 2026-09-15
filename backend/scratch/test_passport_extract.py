import re
from app.modules.ocr.engine import run_ocr
from app.utils.image_utils import bytes_to_bgr
from pathlib import Path

img_path = Path(r"d:\Projects\SIH26\SIH-2026\WhatsApp Image 2026-09-12 at 11.28.17 PM.jpeg")
bgr = bytes_to_bgr(img_path.read_bytes())
ocr_res = run_ocr(bgr)

raw_text = ocr_res.raw_text

def extract_passport_fields(raw_text: str) -> dict:
    fields = {}
    
    # 1. Passport Number
    m_pass = re.search(r"(?:Passport\s*No|PassportNo|Passport\s*#)[\s.:/]*([A-Z0-9]{7,10})", raw_text, re.IGNORECASE)
    if m_pass:
        fields["passport_number"] = m_pass.group(1).upper()
        
    # 2. Surname
    m_sur = re.search(r"Surname[\s.:/]*([A-Z\s]{2,30})", raw_text, re.IGNORECASE)
    if m_sur:
        # stop before Given Name / Date / next label
        val = re.split(r"(?:Given|Date|Sex|Place|\n)", m_sur.group(1), flags=re.IGNORECASE)[0].strip()
        fields["surname"] = val
        
    # 3. Given Name(s)
    m_given = re.search(r"Given\s*Name\(?s\)?[\s.:/]*([A-Z\s]{2,40})", raw_text, re.IGNORECASE)
    if m_given:
        val = re.split(r"(?:Date|Sex|Place|Nationality|\n)", m_given.group(1), flags=re.IGNORECASE)[0].strip()
        fields["given_names"] = val
        
    # Full Name for validation matching
    sur = fields.get("surname", "")
    given = fields.get("given_names", "")
    if sur and given:
        fields["name"] = f"{given} {sur}"
    elif given:
        fields["name"] = given
    elif sur:
        fields["name"] = sur
        
    # Dates: DD/MM/YYYY
    dates = re.findall(r"\b(\d{2}/\d{2}/\d{4})\b", raw_text)
    if len(dates) >= 3:
        fields["date_of_birth"] = dates[0]
        fields["date_of_issue"] = dates[1]
        fields["date_of_expiry"] = dates[2]
    elif len(dates) == 2:
        fields["date_of_birth"] = dates[0]
        fields["date_of_expiry"] = dates[1]
        
    return fields

print("Extracted passport visual fields:", extract_passport_fields(raw_text))
