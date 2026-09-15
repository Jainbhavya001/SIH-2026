import re

def clean_mrz_line1(raw: str) -> str:
    # Remove spaces and non-MRZ chars
    cleaned = re.sub(r"[^A-Z0-9<]", "", raw.upper())
    # Fix <K< or <C< misread for << separator
    cleaned = re.sub(r"<[KC]<", "<<", cleaned)
    # Fix trailing K runs misread for <
    # In TD3 line 1, after the name, everything to col 44 is '<'
    if "<<" in cleaned:
        prefix, _, given_and_fillers = cleaned.partition("<<")
        # given name consists of A-Z and single '<' between names
        # trailing section is mostly '<' and OCR misread 'K'
        m = re.match(r"^([A-Z<]+?)(?:[<K]{3,}|$)", given_and_fillers)
        if m:
            given = m.group(1).replace("K", "<")
            cleaned = prefix + "<<" + given
    # Pad with '<' to 44
    return (cleaned + "<" * 44)[:44]

def clean_mrz_line2(raw: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9<]", "", raw.upper())
    # Fix common 1ND -> IND for country code at pos 10:13
    if len(cleaned) >= 13 and cleaned[10:13] == "1ND":
        cleaned = cleaned[:10] + "IND" + cleaned[13:]
    return (cleaned + "<" * 44)[:44]

line1_raw = "P<INDSRIVASTAVA<K<ANIMESH<KUMARS<<<<K< KKK KKK"
line2_raw = "X9252872<41ND0510305M3306194307 8068004423<98"

l1 = clean_mrz_line1(line1_raw)
l2 = clean_mrz_line2(line2_raw)

print("Cleaned Line 1 (len", len(l1), "):", l1)
print("Cleaned Line 2 (len", len(l2), "):", l2)

from app.modules.ocr.mrz_parser import compute_check_digit, parse_mrz

res = parse_mrz(l1 + "\n" + l2)
print("\nMRZ Result:")
print("Detected:", res.detected)
print("Issuing Country:", res.issuing_country)
print("Surname:", res.surname)
print("Given Names:", res.given_names)
print("Nationality:", res.nationality)
print("Passport No:", res.fields[0].value, "Valid:", res.fields[0].valid)
print("DOB:", res.date_of_birth, "Valid:", res.fields[1].valid)
print("Expiry:", res.date_of_expiry, "Valid:", res.fields[2].valid)
print("Composite Valid:", res.composite_valid)
