"""
Smoke test against a running server (uvicorn app.main:app --port 8000).
Skipped automatically when no server is listening, so `pytest tests/`
stays green in CI; run it directly with `python tests/test_live.py`.
"""
import io

import httpx
import pytest
from PIL import Image, ImageDraw

LIVE_URL = "http://127.0.0.1:8000"


def _server_up() -> bool:
    try:
        return httpx.get(f"{LIVE_URL}/api/health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def _synthetic_passport_png() -> bytes:
    img = Image.new("RGB", (900, 600), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), "REPUBLIC OF UTOPIA - PASSPORT", (0, 0, 0))
    draw.text((30, 480), "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", (0, 0, 0))
    draw.text((30, 520), "L898902C36UTO7408122F1204159ZE184226B<<<<<10", (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_live():
    if not _server_up():
        pytest.skip(f"No server running at {LIVE_URL}")

    with httpx.Client(timeout=120.0) as client:
        r = client.post(
            f"{LIVE_URL}/api/documents/scan",
            data={"document_type": "passport"},
            files={"document": ("passport.png", _synthetic_passport_png(), "image/png")},
        )
    assert r.status_code == 200
    data = r.json()
    print("Scan ID:", data.get("id"))
    print("Risk Verdict:", data["risk"]["verdict"], "Risk Score:", data["risk"]["risk_score"])

    localization = data["localization"]
    assert localization is not None
    assert localization["image_width"] == 900 and localization["image_height"] == 600
    print("Localization backend:", localization["backend"], "model:", localization["model_version"])
    for region in localization["regions"]:
        bbox = region["bbox"]
        assert bbox["xmin"] <= bbox["xmax"] and bbox["ymin"] <= bbox["ymax"]
        print(f"  {region['label']}: ({bbox['xmin']:.0f},{bbox['ymin']:.0f})-"
              f"({bbox['xmax']:.0f},{bbox['ymax']:.0f}) conf={bbox['confidence']:.2f}")


if __name__ == "__main__":
    test_live()
