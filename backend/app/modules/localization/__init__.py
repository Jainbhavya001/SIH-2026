"""
Module 0 — Document Region Localization (Faster R-CNN).

Upstream localization stage that detects semantically meaningful regions
of identity documents before OCR, MRZ parsing, face detection, and
tampering analysis.
"""
from __future__ import annotations

from .config import LocalizationConfig, get_localization_config
from .detector import (
    DocumentRegionDetector,
    crop_region,
    detect_document_regions,
    get_detector,
    trusted_regions,
)
from .synthetic_generator import SyntheticDocumentGenerator

__all__ = [
    "DocumentRegionDetector",
    "LocalizationConfig",
    "SyntheticDocumentGenerator",
    "crop_region",
    "detect_document_regions",
    "get_detector",
    "get_localization_config",
    "trusted_regions",
]
