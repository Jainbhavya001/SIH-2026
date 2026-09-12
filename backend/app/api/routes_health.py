"""System status endpoint — the frontend uses this to show which engines
are live (Tesseract installed? strong face backend available?) so a demo
never silently degrades without the officer knowing."""
from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.modules.ocr.engine import tesseract_binary_available
from app.modules.face.verifier import _STRONG_BACKEND

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health():
    settings = get_settings()
    return {
        "status": "ok",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "engines": {
            "ocr": "tesseract" if tesseract_binary_available() else "unavailable",
            "face_verification": "face_recognition (dlib)" if _STRONG_BACKEND else "histogram+ORB (lightweight)",
        },
    }
