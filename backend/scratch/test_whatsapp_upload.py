import httpx
from pathlib import Path

img_path = Path(r"d:\Projects\SIH26\SIH-2026\WhatsApp Image 2026-09-12 at 11.28.17 PM.jpeg")
img_bytes = img_path.read_bytes()

with httpx.Client() as client:
    r = client.post(
        "http://127.0.0.1:8000/api/documents/scan",
        data={"document_type": "passport"},
        files={"document": ("whatsapp_passport.jpg", img_bytes, "image/jpeg")}
    )
    print("Status code:", r.status_code)
    data = r.json()
    print("Scan ID:", data.get("id"))
    print("Document Type:", data.get("document_type"))
    print("MRZ Summary:", data.get("mrz"))
    print("Validation Summary:", data.get("validation"))
    print("Risk Verdict:", data.get("risk", {}).get("verdict"))
    print("Risk Score:", data.get("risk", {}).get("risk_score"))
