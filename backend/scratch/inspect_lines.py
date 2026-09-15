import re
from app.modules.ocr.engine import run_ocr
from app.utils.image_utils import bytes_to_bgr
from pathlib import Path

img_path = Path(r"d:\Projects\SIH26\SIH-2026\WhatsApp Image 2026-09-12 at 11.28.17 PM.jpeg")
bgr = bytes_to_bgr(img_path.read_bytes())
ocr_res = run_ocr(bgr)

print("--- RAW OCR OUTPUT ---")
lines = [l.strip() for l in ocr_res.raw_text.splitlines() if l.strip()]
for i, line in enumerate(lines):
    print(f"{i}: {line}")
