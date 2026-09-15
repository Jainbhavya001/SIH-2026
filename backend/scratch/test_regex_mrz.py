import re
from app.modules.ocr.engine import run_ocr
from app.utils.image_utils import bytes_to_bgr
from pathlib import Path

img_path = Path(r"d:\Projects\SIH26\SIH-2026\WhatsApp Image 2026-09-12 at 11.28.17 PM.jpeg")
bgr = bytes_to_bgr(img_path.read_bytes())
ocr_res = run_ocr(bgr)
raw_text = ocr_res.raw_text

m = re.search(r"(P<[A-Z0-9<K\s]{30,80}?)\s+([A-Z0-9]{8,10}[A-Z0-9<K\s]{30,80})", raw_text)
if m:
    print("Match Line 1 candidate:", repr(m.group(1)))
    print("Match Line 2 candidate:", repr(m.group(2)))
else:
    print("No match")
