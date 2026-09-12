"""
Module 1 — OCR Extraction.

Wraps whichever OCR backend is installed behind a stable interface so the
rest of the system never imports `pytesseract` directly. If Tesseract's
binary isn't installed on the host, we fail soft with a clear error the
frontend can surface, instead of crashing the whole scan.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

try:
    import pytesseract
    from pytesseract import Output

    _TESSERACT_PACKAGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TESSERACT_PACKAGE_AVAILABLE = False


def tesseract_binary_available() -> bool:
    """
    True only if the actual Tesseract *executable* is reachable — having the
    `pytesseract` Python package installed is necessary but not sufficient,
    since it's just a thin wrapper that shells out to the binary. Used by
    the health endpoint so the frontend's status badge reflects reality
    instead of a package import that always succeeds after `pip install`.
    """
    if not _TESSERACT_PACKAGE_AVAILABLE:
        return False
    _configure_tesseract()
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# Back-compat alias used elsewhere in this module for the soft-fail path below.
_TESSERACT_AVAILABLE = _TESSERACT_PACKAGE_AVAILABLE


@dataclass
class OCRWord:
    text: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h


@dataclass
class OCRResult:
    raw_text: str
    words: list[OCRWord] = field(default_factory=list)
    mean_confidence: float = 0.0
    engine_available: bool = True
    warning: str | None = None


def _configure_tesseract() -> None:
    settings = get_settings()
    if settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


def run_ocr(image: np.ndarray) -> OCRResult:
    """
    Run OCR over a BGR (OpenCV-style) image array and return raw text plus
    word-level boxes/confidences.

    Design note: this function is intentionally the *only* place that talks
    to the OCR backend. Swapping Tesseract for a cloud OCR/vision-LLM later
    means editing only this file.
    """
    if not _TESSERACT_AVAILABLE:
        return OCRResult(
            raw_text="",
            engine_available=False,
            warning=(
                "Tesseract OCR is not installed on this machine. Install the "
                "Tesseract binary and the `pytesseract` package (see "
                "backend/README.md) to enable real text extraction. "
                "Downstream modules will run in degraded mode."
            ),
        )

    _configure_tesseract()

    try:
        data = pytesseract.image_to_data(
            image, output_type=Output.DICT, config="--oem 3 --psm 6"
        )
    except pytesseract.TesseractNotFoundError:
        return OCRResult(
            raw_text="",
            engine_available=False,
            warning=(
                "Tesseract binary not found on PATH. Set TESSERACT_CMD in "
                "your .env to the full path of tesseract.exe."
            ),
        )

    words: list[OCRWord] = []
    confidences: list[float] = []
    text_parts: list[str] = []

    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = float(data["conf"][i]) if data["conf"][i] != "-1" else 0.0
        box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        words.append(OCRWord(text=text, confidence=conf, box=box))
        confidences.append(conf)
        text_parts.append(text)

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return OCRResult(
        raw_text=" ".join(text_parts),
        words=words,
        mean_confidence=round(mean_conf, 2),
    )
