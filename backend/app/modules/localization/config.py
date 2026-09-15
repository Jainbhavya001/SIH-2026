"""
Configuration for Module 0 — Document Region Localization.

Environment-driven like `app/config.py`, but kept next to the detector
because nothing outside this module needs the model internals (class list,
per-class thresholds, architecture name). Every value can be overridden
without code changes, which matters for switching between a GPU demo box
and a CPU-only CI runner.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import BACKEND_ROOT

# Class index order is the contract with trained weights: index 0 is the
# implicit background class required by torchvision's Faster R-CNN. Labels
# match what `synthetic_generator.py` annotates.
REGION_CLASSES: list[str] = [
    "background",
    "photo",
    "name",
    "surname",
    "date_of_birth",
    "date_of_issue",
    "date_of_expiry",
    "document_number",
    "nationality",
    "mrz",
    "signature",
    "stamp",
]

# Labels emitted only by the CPU fallbacks. They describe layout, not
# semantics, so downstream modules treat them as coarse hints.
FALLBACK_LABELS: list[str] = ["document", "visual_zone"]

TEXT_FIELD_LABELS: frozenset[str] = frozenset({
    "name",
    "surname",
    "date_of_birth",
    "date_of_issue",
    "date_of_expiry",
    "document_number",
    "nationality",
})

# Regions where an alteration changes who the document identifies — the
# tampering engine inspects these individually.
HIGH_RISK_LABELS: frozenset[str] = frozenset({
    "photo",
    "name",
    "surname",
    "date_of_birth",
    "date_of_expiry",
    "document_number",
    "mrz",
})

DOCUMENT_TYPE_REGIONS: dict[str, list[str]] = {
    "passport": [
        "photo", "name", "surname", "date_of_birth", "date_of_issue",
        "date_of_expiry", "document_number", "nationality", "mrz", "signature",
    ],
    "visa": [
        "photo", "name", "surname", "date_of_birth", "date_of_issue",
        "date_of_expiry", "document_number", "nationality", "stamp",
    ],
    "national_id": [
        "photo", "name", "surname", "date_of_birth", "document_number", "nationality",
    ],
    "driving_license": [
        "photo", "name", "surname", "date_of_birth", "date_of_issue",
        "date_of_expiry", "document_number", "nationality",
    ],
    "permit": [
        "photo", "name", "date_of_birth", "date_of_issue", "date_of_expiry", "document_number",
    ],
    "pan_card": [
        "photo", "name", "surname", "date_of_birth", "document_number", "signature",
    ],
    "aadhaar": [
        "photo", "name", "date_of_birth", "document_number",
    ],
}

# Regions whose absence on a document type is itself a red flag.
CRITICAL_REGIONS: dict[str, list[str]] = {
    "passport": ["photo", "mrz"],
    "visa": ["photo"],
    "national_id": ["photo"],
    "driving_license": ["photo"],
    "permit": [],
    "pan_card": ["photo", "document_number"],
    "aadhaar": ["photo"],
}

DEFAULT_CLASS_THRESHOLDS: dict[str, float] = {
    "photo": 0.5,
    "name": 0.4,
    "surname": 0.4,
    "date_of_birth": 0.4,
    "date_of_issue": 0.4,
    "date_of_expiry": 0.4,
    "document_number": 0.4,
    "nationality": 0.4,
    "mrz": 0.5,
    "signature": 0.3,
    "stamp": 0.4,
}

SUPPORTED_ARCHITECTURES = (
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn",
)

FALLBACK_MODES = ("heuristic", "full_image", "none")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LocalizationConfig:
    enabled: bool = True
    architecture: str = "fasterrcnn_resnet50_fpn"
    model_path: Path = BACKEND_ROOT / "models" / "localization" / "fasterrcnn_document_regions.pth"
    model_version: str = "fasterrcnn_resnet50_fpn-docregions-v1"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:N"
    confidence_threshold: float = 0.3
    class_thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CLASS_THRESHOLDS))
    class_names: list[str] = field(default_factory=lambda: list(REGION_CLASSES))
    max_detections_per_label: int = 3
    # Fallback used when torch/torchvision or the weights file is missing.
    fallback_mode: str = "heuristic"
    # Randomly initialised heads produce meaningless boxes; only allow them
    # for smoke-testing the torch code path.
    allow_untrained: bool = False
    # Minimum region confidence before a crop is trusted for OCR/face work.
    min_region_confidence: float = 0.3
    save_crops: bool = False
    crops_dir: Path = BACKEND_ROOT / "uploads" / "crops"

    def threshold_for(self, label: str) -> float:
        return max(self.confidence_threshold, self.class_thresholds.get(label, 0.0))


@lru_cache
def get_localization_config() -> LocalizationConfig:
    architecture = os.getenv("LOCALIZATION_ARCHITECTURE", "fasterrcnn_resnet50_fpn")
    if architecture not in SUPPORTED_ARCHITECTURES:
        architecture = "fasterrcnn_resnet50_fpn"

    fallback_mode = os.getenv("LOCALIZATION_FALLBACK", "heuristic").strip().lower()
    if fallback_mode not in FALLBACK_MODES:
        fallback_mode = "heuristic"

    model_path_env = os.getenv("LOCALIZATION_MODEL_PATH")
    defaults = LocalizationConfig()

    return LocalizationConfig(
        enabled=_env_bool("LOCALIZATION_ENABLED", True),
        architecture=architecture,
        model_path=Path(model_path_env) if model_path_env else defaults.model_path,
        model_version=os.getenv("LOCALIZATION_MODEL_VERSION", f"{architecture}-docregions-v1"),
        device=os.getenv("LOCALIZATION_DEVICE", "auto"),
        confidence_threshold=float(os.getenv("LOCALIZATION_CONFIDENCE_THRESHOLD", "0.3")),
        max_detections_per_label=int(os.getenv("LOCALIZATION_MAX_PER_LABEL", "3")),
        fallback_mode=fallback_mode,
        allow_untrained=_env_bool("LOCALIZATION_ALLOW_UNTRAINED", False),
        min_region_confidence=float(os.getenv("LOCALIZATION_MIN_REGION_CONFIDENCE", "0.3")),
        save_crops=_env_bool("LOCALIZATION_SAVE_CROPS", False),
    )
