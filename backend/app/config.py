"""
Centralized, environment-driven configuration.

Everything that might change between a dev laptop, a CI runner, and a demo
server lives here — never hardcode paths/thresholds inside a module.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings:
    # --- General ---
    APP_NAME: str = "AI Document Screening System"
    APP_VERSION: str = "0.1.0"
    ENV: str = os.getenv("APP_ENV", "development")

    # --- CORS ---
    ALLOWED_ORIGINS: list[str] = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")

    # --- Storage ---
    UPLOAD_DIR: Path = BACKEND_ROOT / "uploads"
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "12"))

    # --- OCR ---
    TESSERACT_CMD: str | None = os.getenv("TESSERACT_CMD")  # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe

    # --- Tampering detection thresholds (tunable without code changes) ---
    ELA_JPEG_QUALITY: int = int(os.getenv("ELA_JPEG_QUALITY", "90"))
    ELA_SUSPICION_THRESHOLD: float = float(os.getenv("ELA_SUSPICION_THRESHOLD", "18.0"))
    COPY_MOVE_MATCH_THRESHOLD: int = int(os.getenv("COPY_MOVE_MATCH_THRESHOLD", "12"))

    # --- Face verification ---
    FACE_MATCH_THRESHOLD: float = float(os.getenv("FACE_MATCH_THRESHOLD", "0.55"))

    # --- Risk engine weights (must sum to 1.0) ---
    WEIGHT_VALIDATION: float = float(os.getenv("WEIGHT_VALIDATION", "0.35"))
    WEIGHT_TAMPERING: float = float(os.getenv("WEIGHT_TAMPERING", "0.40"))
    WEIGHT_FACE: float = float(os.getenv("WEIGHT_FACE", "0.25"))

    RISK_CLEAR_MAX: int = int(os.getenv("RISK_CLEAR_MAX", "30"))
    RISK_REVIEW_MAX: int = int(os.getenv("RISK_REVIEW_MAX", "65"))


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return settings
