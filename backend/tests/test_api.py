"""
Integration tests for FastAPI endpoints:
- GET /
- GET /api/health
- POST /api/documents/scan
- GET /api/documents/scans
- GET /api/documents/scans/{scan_id}
- GET /api/documents/stats
- POST /api/face/verify
"""
import io
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.models.schemas import ScanResponse

client = TestClient(app)


def _create_dummy_image() -> bytes:
    img = Image.new("RGB", (200, 200), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "message" in data
    assert data["docs"] == "/docs"


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "engines" in data


def test_scan_document_and_retrieval():
    img_bytes = _create_dummy_image()
    files = {"document": ("doc.png", img_bytes, "image/png")}
    data = {"document_type": "passport"}

    res = client.post("/api/documents/scan", data=data, files=files)
    assert res.status_code == 200
    scan_data = res.json()
    assert "id" in scan_data
    scan_id = scan_data["id"]
    assert scan_data["document_type"] == "passport"
    assert "risk" in scan_data

    # Region localization runs upstream and is reported alongside the
    # existing sections (fallback backend on CPU-only test machines).
    localization = scan_data["localization"]
    assert localization is not None
    assert localization["backend"] in {"faster_rcnn", "heuristic", "full_image", "none"}
    assert localization["image_width"] == 200 and localization["image_height"] == 200
    assert localization["processing_time_ms"] >= 0
    assert isinstance(localization["model_version"], str)
    for region in localization["regions"]:
        bbox = region["bbox"]
        assert 0 <= bbox["xmin"] <= bbox["xmax"] <= 200
        assert 0 <= bbox["ymin"] <= bbox["ymax"] <= 200
        assert 0 <= bbox["confidence"] <= 1
    assert isinstance(scan_data["tampering"]["region_findings"], list)
    assert "region_text" in scan_data["ocr"]
    assert "localization_completeness" in scan_data["risk"]

    # Test list scans
    res_list = client.get("/api/documents/scans")
    assert res_list.status_code == 200
    scans = res_list.json()
    assert any(s["id"] == scan_id for s in scans)

    # Test get scan by id
    res_get = client.get(f"/api/documents/scans/{scan_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == scan_id
    assert res_get.json()["localization"] == localization

    # Test stats
    res_stats = client.get("/api/documents/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert stats["total_scans"] >= 1


def test_scan_non_mrz_document_reports_localization():
    files = {"document": ("id.png", _create_dummy_image(), "image/png")}
    res = client.post("/api/documents/scan", data={"document_type": "national_id"}, files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["mrz"] is None
    assert data["localization"] is not None
    assert all(r["label"] != "mrz" for r in data["localization"]["regions"])


def test_health_reports_localization_backend():
    data = client.get("/api/health").json()
    assert "localization" in data["engines"]
    assert data["localization"]["fallback_mode"] in {"heuristic", "full_image", "none"}


def test_scan_records_without_localization_still_deserialize():
    """Scans stored before Module 0 existed have no localization/region keys."""
    legacy = {
        "id": "legacy", "timestamp": "2026-01-01T00:00:00Z", "document_type": "visa",
        "ocr": {"raw_text": "", "mean_confidence": 0.0, "engine_available": False, "extracted_fields": {}},
        "mrz": None,
        "validation": {"score": 100, "issues": [], "checks_run": 1},
        "tampering": {"tampering_score": 0, "verdict": "clean", "evidence": [], "ela_suspicious": False,
                      "ela_max_error": 0.0, "copy_move_matches": 0},
        "face": {"attempted": False},
        "risk": {"risk_score": 0, "verdict": "CLEAR", "contributing_factors": [], "validation_score": 100,
                 "tampering_score": 0},
    }
    parsed = ScanResponse(**legacy)
    assert parsed.localization is None
    assert parsed.tampering.region_findings == []


def test_face_verify_endpoint():
    img1 = _create_dummy_image()
    img2 = _create_dummy_image()
    files = {
        "document_photo": ("doc.png", img1, "image/png"),
        "live_photo": ("live.png", img2, "image/png"),
    }
    res = client.post("/api/face/verify", files=files)
    assert res.status_code == 200
    face_data = res.json()
    assert "similarity" in face_data
    assert "is_match" in face_data
