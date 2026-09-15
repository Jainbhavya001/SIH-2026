"""
Centralized, environment-driven configuration.

Everything that might change between a dev laptop, a CI runner, and a demo
server lives here — never hardcode paths/thresholds inside a module.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env")


def _get_tesseract_cmd() -> str | None:
    cmd = os.getenv("TESSERACT_CMD")
    if cmd and os.path.exists(cmd):
        return cmd
    if os.name == "nt":
        default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_win_path):
            return default_win_path
    return cmd


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
    TESSERACT_CMD: str | None = _get_tesseract_cmd()

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

    # --- Region localization contribution (additive, only from a trained model) ---
    LOCALIZATION_RISK_MAX_POINTS: int = int(os.getenv("LOCALIZATION_RISK_MAX_POINTS", "15"))
    LOCALIZATION_MISSING_CRITICAL_POINTS: int = int(os.getenv("LOCALIZATION_MISSING_CRITICAL_POINTS", "8"))
    LOCALIZATION_LOW_CONFIDENCE: float = float(os.getenv("LOCALIZATION_LOW_CONFIDENCE", "0.5"))

    RISK_CLEAR_MAX: int = int(os.getenv("RISK_CLEAR_MAX", "30"))
    RISK_REVIEW_MAX: int = int(os.getenv("RISK_REVIEW_MAX", "65"))


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return settings
