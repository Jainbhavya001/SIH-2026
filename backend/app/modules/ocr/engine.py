"""
Module 1 — OCR Extraction.

Wraps whichever OCR backend is installed behind a stable interface so the
rest of the system never imports `pytesseract` directly. Supports:
  1. PaddleOCR (paddlepaddle) — modern, layout-aware, better on structured docs
  2. Tesseract (pytesseract) — classic, works if the binary is installed

If neither backend is available, we fail soft with a clear error the
frontend can surface, instead of crashing the whole scan.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal

import cv2
import numpy as np

from app.config import get_settings
from app.models.schemas import DetectedRegion

logger = logging.getLogger(__name__)

# ── Try PaddleOCR first ───────────────────────────────────────────────
_PADDLE_AVAILABLE = False
_paddle_ocr = None

try:
    import paddlepaddle  # noqa: F401  – populates paddlepaddle.__version__
    from paddleocr import PaddleOCR  # type: ignore

    _paddle_ocr = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
    _PADDLE_AVAILABLE = True
    logger.info("PaddleOCR backend available.")
except ImportError:  # pragma: no cover
    logger.debug("PaddleOCR not available (paddlepaddle not installed).")
except Exception as e:  # pragma: no cover
    logger.debug(f"PaddleOCR init failed: {e}")


# ── Tesseract fallback ─────────────────────────────────────────────────
_TESSERACT_PACKAGE_AVAILABLE = False
_tesseract_available = False

try:
    import pytesseract
    from pytesseract import Output

    _TESSERACT_PACKAGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

def _configure_tesseract():
    if _TESSERACT_PACKAGE_AVAILABLE:
        settings = get_settings()
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

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


# Back-compat alias.
_TESSERACT_AVAILABLE = _TESSERACT_PACKAGE_AVAILABLE


# ── Shared output types ────────────────────────────────────────────────

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
    backend: str = "tesseract"  # "paddle" or "tesseract"


# ── Backend helpers ─────────────────────────────────────────────────────

def _run_paddle_ocr(image: np.ndarray) -> OCRResult:
    """Run PaddleOCR on a BGR image; map its output to OCRResult."""
    if _paddle_ocr is None:
        return OCRResult(
            raw_text="",
            engine_available=False,
            warning="PaddleOCR backend initialised but engine instance is None.",
            backend="paddle",
        )

    try:
        # PaddleOCR returns list of lines, each line is list of [[bbox, text, score], ...]
        results = _paddle_ocr.ocr(image, cls=True)

        if not results or results is None:
            return OCRResult(
                raw_text="",
                engine_available=False,
                warning="PaddleOCR produced no results.",
                backend="paddle",
            )

        # PaddleOCR may return a list per page; take the first page.
        page_results = results[0] if isinstance(results, list) and results else []

        words: list[OCRWord] = []
        text_parts: list[str] = []
        confidences: list[float] = []

        for line in page_results:
            # line format: [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, conf), ... ]
            if isinstance(line, list) and len(line) >= 2:
                box_coords = line[0]
                text_line = line[1][0] if len(line[1]) > 0 else ""
                conf_line = line[1][1] if len(line[1]) > 1 else 0.0

                # Normalise bbox to (x, y, w, h)
                xs = [p[0] for p in box_coords]
                ys = [p[1] for p in box_coords]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                box = (int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin))

                # Handle confidence: Paddle may return None or a numeric string
                try:
                    conf = float(conf_line) if conf_line is not None else 0.0
                except (ValueError, TypeError):
                    conf = 0.0

                if text_line.strip():
                    words.append(OCRWord(text=text_line.strip(), confidence=conf, box=box))
                    confidences.append(conf)
                    text_parts.append(text_line.strip())

        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

        raw_text = "\n".join(text_parts) if text_parts else ""

        return OCRResult(
            raw_text=raw_text,
            words=words,
            mean_confidence=round(mean_conf, 2),
            engine_available=True,
            backend="paddle",
        )

    except Exception as e:
        logger.error(f"PaddleOCR runtime error: {e}")
        return OCRResult(
            raw_text="",
            engine_available=False,
            warning=f"PaddleOCR runtime error: {e}",
            backend="paddle",
        )


def _run_tesseract_ocr(image: np.ndarray, config: str = "--oem 3 --psm 6") -> OCRResult:
    """Run Tesseract OCR on a BGR image; map its output to OCRResult."""
    if not _TESSERACT_PACKAGE_AVAILABLE:
        return OCRResult(
            raw_text="",
            engine_available=False,
            warning=(
                "Tesseract OCR package not installed. Run: pip install pytesseract "
                "(and install the Tesseract binary separately, see README)."
            ),
            backend="tesseract",
        )

    _configure_tesseract()

    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT, config=config)
    except pytesseract.TesseractNotFoundError:
        return OCRResult(
            raw_text="",
            engine_available=False,
            warning=(
                "Tesseract binary not found on PATH. Set TESSERACT_CMD in "
                "your .env to the full path of tesseract.exe."
            ),
            backend="tesseract",
        )

    words: list[OCRWord] = []
    confidences: list[float] = []
    lines_map: dict[tuple[int, int], list[str]] = {}

    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = float(data["conf"][i]) if data["conf"][i] != "-1" else 0.0
        box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        words.append(OCRWord(text=text, confidence=conf, box=box))
        confidences.append(conf)

        block_line_key = (data["block_num"][i], data["line_num"][i])
        lines_map.setdefault(block_line_key, []).append(text)

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    line_strings = [" ".join(w) for w in lines_map.values()]
    raw_text = "\n".join(line_strings)

    return OCRResult(
        raw_text=raw_text,
        words=words,
        mean_confidence=round(mean_conf, 2),
        engine_available=True,
        backend="tesseract",
    )


OCRProfile = Literal["document", "line", "mrz"]

_MRZ_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

# Tesseract page-segmentation per kind of input: a whole page (psm 6), a
# single localized field value (psm 7), or the MRZ band restricted to the
# ICAO 9303 OCR-B character set.
_TESSERACT_CONFIGS: dict[str, str] = {
    "document": "--oem 3 --psm 6",
    "line": "--oem 3 --psm 7",
    "mrz": f"--oem 3 --psm 6 -c tessedit_char_whitelist={_MRZ_CHARSET}",
}

# Target crop heights before OCR: Tesseract is most accurate around 30-40px
# per character row, and localized crops from phone photos are often smaller.
_TARGET_HEIGHT = {"line": 64, "mrz": 140, "document": 0}


def run_ocr(image: np.ndarray, profile: OCRProfile = "document") -> OCRResult:
    """
    Run OCR over a BGR (OpenCV-style) image array and return raw text plus
    word-level boxes/confidences.

    `profile` tunes the engine for the input: "document" for a full page
    (the default, unchanged behaviour), "line" for a single localized field
    crop, and "mrz" for a localized machine-readable zone.

    Priority: PaddleOCR → Tesseract → graceful degradation.
    The returned OCRResult has the same shape regardless of backend,
    so downstream modules (MRZ parser, field extractors, validation)
    require zero changes.
    """
    # 1) Try PaddleOCR first (layout-aware, so profiles need no special config)
    if _PADDLE_AVAILABLE and _paddle_ocr is not None:
        result = _run_paddle_ocr(image)
        if result.engine_available:
            logger.info("PaddleOCR succeeded.")
            return result
        else:
            logger.warning("PaddleOCR failed; falling back to Tesseract.")

    # 2) Fall back to Tesseract
    if _TESSERACT_AVAILABLE:
        result = _run_tesseract_ocr(image, config=_TESSERACT_CONFIGS.get(profile, _TESSERACT_CONFIGS["document"]))
        if result.engine_available:
            logger.info("Tesseract succeeded.")
            return result
        else:
            logger.warning("Tesseract reported unavailable; returning degraded result.")

    # 3) Both backends unavailable — return degraded result
    return OCRResult(
        raw_text="",
        engine_available=False,
        warning=(
            "Neither PaddleOCR nor Tesseract OCR is available on this machine. "
            "Install one of them to enable real text extraction. "
            "• PaddleOCR: pip install paddlepaddle + paddlepaddleocr, then ensure "
            "the Tesseract binary is on PATH (for Tesseract fallback), or "
            "• Tesseract: install the Tesseract binary and pip install pytesseract "
            "(see backend/README.md for OCR setup notes). "
            "Downstream modules will run in degraded mode."
        ),
        backend="none",
    )


# ── Region-targeted OCR (fed by Module 0 localization) ─────────────────

@dataclass
class RegionOCRResult:
    label: str
    text: str
    confidence: float  # OCR mean confidence for the crop
    region_confidence: float  # localization confidence of the source box
    box: tuple[int, int, int, int]  # x, y, w, h in full-image pixels
    backend: str


def prepare_region_for_ocr(crop: np.ndarray, profile: OCRProfile = "line") -> np.ndarray:
    """Pad a tight crop with background and upscale it to an OCR-friendly
    height. Characters touching the crop edge are a common Tesseract failure."""
    if crop.size == 0:
        return crop
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    target = _TARGET_HEIGHT.get(profile, 0)
    h = crop.shape[0]
    if target and h < target:
        scale = target / float(h)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    border = max(8, crop.shape[0] // 8)
    return cv2.copyMakeBorder(crop, border, border, border, border, cv2.BORDER_REPLICATE)


def _profile_for_label(label: str) -> OCRProfile:
    if label == "mrz":
        return "mrz"
    if label in {"name", "surname", "date_of_birth", "date_of_issue", "date_of_expiry",
                 "document_number", "nationality"}:
        return "line"
    return "document"


def run_region_ocr(
    image: np.ndarray,
    regions: Iterable[DetectedRegion],
    labels: Iterable[str] | None = None,
) -> dict[str, RegionOCRResult]:
    """
    OCR the highest-confidence localized region for each label. Returns
    label -> result for crops that produced text. Stops early if no OCR
    engine is available so a missing binary isn't retried once per region.
    """
    from app.modules.localization.detector import bbox_to_xywh, crop_region

    wanted = set(labels) if labels is not None else None
    best: dict[str, DetectedRegion] = {}
    for region in regions:
        if wanted is not None and region.label not in wanted:
            continue
        current = best.get(region.label)
        if current is None or region.bbox.confidence > current.bbox.confidence:
            best[region.label] = region

    results: dict[str, RegionOCRResult] = {}
    for label, region in best.items():
        profile = _profile_for_label(label)
        crop = crop_region(image, region.bbox, padding=2, padding_ratio=0.04)
        if crop.size == 0 or min(crop.shape[:2]) < 6:
            continue
        ocr = run_ocr(prepare_region_for_ocr(crop, profile), profile=profile)
        if not ocr.engine_available:
            logger.info("Region OCR skipped: no OCR engine available.")
            break
        text = ocr.raw_text.strip()
        if not text:
            continue
        results[label] = RegionOCRResult(
            label=label,
            text=text,
            confidence=ocr.mean_confidence,
            region_confidence=region.bbox.confidence,
            box=bbox_to_xywh(region.bbox),
            backend=ocr.backend,
        )
    return results
