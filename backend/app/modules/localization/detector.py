"""
Module 0 — Document Region Localization (Faster R-CNN).

Runs BEFORE OCR, MRZ parsing, face detection, and tampering analysis and
tells each of them where to look:

  - text field regions (name, surname, dates, document number, nationality)
    are cropped and OCR'd individually,
  - the MRZ region is OCR'd with an MRZ-only character set before parsing,
  - the photo region narrows the face search,
  - high-risk regions are inspected individually for tampering.

Backends, chosen automatically:

1. Faster R-CNN (torchvision) — used when torch/torchvision are installed
   and a trained weights file exists at `LOCALIZATION_MODEL_PATH`.
2. Heuristic fallback (OpenCV only) — document boundary, MRZ band via
   black-hat morphology, and photo region via face detection. Emits low
   confidence boxes so downstream modules treat them as hints.
3. Full-image fallback — one "document" box covering the whole image.

The fallbacks keep the pipeline and the test suite running on CPU-only
machines without multi-hundred-megabyte weight files.
"""
from __future__ import annotations

import importlib.util
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import BACKEND_ROOT
from app.models.schemas import BoundingBox, DetectedRegion, RegionLocalizationResult
from app.modules.face.detector import detect_faces
from app.modules.localization.config import (
    CRITICAL_REGIONS,
    DOCUMENT_TYPE_REGIONS,
    FALLBACK_LABELS,
    LocalizationConfig,
    get_localization_config,
)

logger = logging.getLogger(__name__)

HEURISTIC_MODEL_VERSION = "heuristic-opencv-v1"
FULL_IMAGE_MODEL_VERSION = "full-image-v1"

_MRZ_DOCUMENT_TYPES = {None, "passport", "visa"}


# ── Geometry helpers ───────────────────────────────────────────────────

def make_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float, confidence: float,
    image_shape: tuple[int, ...] | None = None,
) -> BoundingBox:
    """Build a BoundingBox with ordered corners, clipped to the image if given."""
    x0, x1 = sorted((float(xmin), float(xmax)))
    y0, y1 = sorted((float(ymin), float(ymax)))
    if image_shape is not None:
        h, w = image_shape[:2]
        x0, x1 = min(max(x0, 0.0), float(w)), min(max(x1, 0.0), float(w))
        y0, y1 = min(max(y0, 0.0), float(h)), min(max(y1, 0.0), float(h))
    return BoundingBox(
        xmin=round(x0, 2), ymin=round(y0, 2), xmax=round(x1, 2), ymax=round(y1, 2),
        confidence=round(float(min(max(confidence, 0.0), 1.0)), 4),
    )


def bbox_to_xywh(bbox: BoundingBox) -> tuple[int, int, int, int]:
    x0, y0 = int(np.floor(bbox.xmin)), int(np.floor(bbox.ymin))
    x1, y1 = int(np.ceil(bbox.xmax)), int(np.ceil(bbox.ymax))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def crop_region(
    image_bgr: np.ndarray,
    bbox: BoundingBox,
    padding: int = 0,
    padding_ratio: float = 0.0,
) -> np.ndarray:
    """
    Crop `bbox` out of an image, optionally padded by a fixed number of
    pixels and/or a fraction of the box size. Coordinates are clipped to the
    image; a box entirely outside the image yields an empty array (check
    `.size`) rather than raising, so callers can fall back gracefully.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("crop_region received an empty image.")
    h, w = image_bgr.shape[:2]
    pad_x = padding + bbox.width * padding_ratio
    pad_y = padding + bbox.height * padding_ratio

    x0 = int(np.floor(max(0.0, bbox.xmin - pad_x)))
    y0 = int(np.floor(max(0.0, bbox.ymin - pad_y)))
    x1 = int(np.ceil(min(float(w), bbox.xmax + pad_x)))
    y1 = int(np.ceil(min(float(h), bbox.ymax + pad_y)))

    if x1 <= x0 or y1 <= y0:
        return image_bgr[0:0, 0:0].copy()
    return image_bgr[y0:y1, x0:x1].copy()


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("detect_regions expects a non-empty numpy image array.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape {image.shape}.")


def _contrast_normalize(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the luminance channel — geometry-preserving, so boxes found
    on the normalised image are valid on the original."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


# ── Heuristic (OpenCV-only) localization ───────────────────────────────

def _find_document_box(image_bgr: np.ndarray) -> BoundingBox:
    """Largest roughly-rectangular contour covering a meaningful share of the
    frame (a document photographed on a table); otherwise the full image."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = float(h * w)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        x, y, cw, ch = cv2.boundingRect(contour)
        coverage = (cw * ch) / image_area
        if coverage > 0.97:
            break  # contour is the frame itself — no separate document edge
        if coverage < 0.35:
            break  # contours are sorted; nothing larger remains
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            return make_bbox(x, y, x + cw, y + ch, 0.5, image_bgr.shape)
    return make_bbox(0, 0, w, h, 0.1, image_bgr.shape)


def _count_text_rows(blackhat_patch: np.ndarray) -> int:
    """Count separated dark-text rows via a horizontal projection profile."""
    if blackhat_patch.size == 0:
        return 0
    _, binary = cv2.threshold(blackhat_patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    profile = (binary > 0).sum(axis=1).astype(np.float32)
    if profile.max() <= 0:
        return 0
    active = profile > 0.15 * profile.max()
    rows, run = 0, 0
    for is_text in active:
        if is_text:
            run += 1
        else:
            rows += run >= 2
            run = 0
    rows += run >= 2
    return rows


def _find_mrz_band(image_bgr: np.ndarray, doc_box: BoundingBox) -> BoundingBox | None:
    """
    Classic MRZ localisation: black-hat morphology reveals dark text on a
    light background, a horizontal gradient plus closing merges the dense
    OCR-B character rows into one wide band, and the band must sit in the
    lower half of the document spanning most of its width.
    """
    dx, dy, dw, dh = bbox_to_xywh(doc_box)
    doc = image_bgr[dy:dy + dh, dx:dx + dw]
    if doc.size == 0 or dw < 50 or dh < 50:
        return None

    scale = 600.0 / dw if dw > 600 else 1.0
    if scale != 1.0:
        doc = cv2.resize(doc, (int(dw * scale), int(dh * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = doc.shape[:2]

    gray = cv2.GaussianBlur(cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
    grad = np.absolute(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1))
    grad_max = float(grad.max())
    if grad_max <= 0:
        return None
    grad = (255 * (grad - grad.min()) / (grad_max - grad.min() + 1e-6)).astype(np.uint8)

    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect_kernel)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, square_kernel)
    thresh = cv2.erode(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if ch == 0:
            continue
        aspect = cw / float(ch)
        width_ratio = cw / float(sw)
        centre_y = (y + ch / 2.0) / sh
        if aspect < 5 or width_ratio < 0.25 or centre_y < 0.55:
            continue
        # Geometry alone also matches ordinary text lines; an MRZ is 2 (TD3)
        # or 3 (TD1) stacked rows of the same length.
        rows = _count_text_rows(blackhat[max(0, y - 4):y + ch + 4, x:x + cw])
        if rows not in (2, 3):
            continue
        score = width_ratio + centre_y  # prefer wide bands near the bottom edge
        if best is None or score > best[0]:
            best = (score, (x, y, cw, ch))

    if best is None:
        return None

    x, y, cw, ch = best[1]
    pad_x, pad_y = int(cw * 0.03), int(ch * 0.15)
    x0 = dx + max(0, x - pad_x) / scale
    y0 = dy + max(0, y - pad_y) / scale
    x1 = dx + min(sw, x + cw + pad_x) / scale
    y1 = dy + min(sh, y + ch + pad_y) / scale
    confidence = 0.3 + 0.15 * min(1.0, cw / float(sw))
    return make_bbox(x0, y0, x1, y1, confidence, image_bgr.shape)


def _find_photo_region(image_bgr: np.ndarray, doc_box: BoundingBox) -> BoundingBox | None:
    """Expand the largest face inside the document to the portrait box that
    typically frames it (head plus shoulders)."""
    faces = [
        f for f in detect_faces(image_bgr)
        if doc_box.xmin <= f.x + f.w / 2 <= doc_box.xmax and doc_box.ymin <= f.y + f.h / 2 <= doc_box.ymax
    ]
    if not faces:
        return None
    face = max(faces, key=lambda f: f.w * f.h)
    return make_bbox(
        face.x - 0.45 * face.w,
        face.y - 0.6 * face.h,
        face.x + 1.45 * face.w,
        face.y + 1.8 * face.h,
        0.45,
        image_bgr.shape,
    )


# ── Detector ────────────────────────────────────────────────────────────

class DocumentRegionDetector:
    """Faster R-CNN document region detector with CPU-safe fallbacks."""

    def __init__(self, config: LocalizationConfig | None = None):
        self.config = config or get_localization_config()
        self.class_names: list[str] = list(self.config.class_names)
        self.model_version: str = self.config.model_version
        self.load_error: str | None = None
        self._model: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._load_attempted = False
        self._lock = threading.Lock()

    # -- model lifecycle ---------------------------------------------------

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        return str(self._device) if self._device is not None else None

    def load(self) -> bool:
        """Load the Faster R-CNN model once. Returns True if it is usable."""
        if self._load_attempted:
            return self._model is not None
        with self._lock:
            if self._load_attempted:
                return self._model is not None
            try:
                self._load_model()
            except Exception as e:  # never let a bad checkpoint take down a scan
                self._model = None
                self.load_error = f"Failed to load Faster R-CNN: {e}"
                logger.exception("Faster R-CNN load failed; localization will use fallback.")
            finally:
                self._load_attempted = True
        return self._model is not None

    def _load_model(self) -> None:
        cfg = self.config
        if not cfg.enabled:
            self.load_error = "Localization disabled (LOCALIZATION_ENABLED=false)."
            return
        try:
            import torch
            from torchvision.models import detection as tv_detection
        except ImportError as e:
            self.load_error = f"torch/torchvision not installed ({e})."
            return

        weights_path = Path(cfg.model_path)
        has_weights = weights_path.is_file()
        if not has_weights and not cfg.allow_untrained:
            self.load_error = f"Faster R-CNN weights not found at {weights_path}."
            return

        device = self._resolve_device(torch, cfg.device)
        state_dict = None
        if has_weights:
            checkpoint = self._torch_load(torch, weights_path, device)
            state_dict = self._unpack_checkpoint(checkpoint)

        builder = getattr(tv_detection, cfg.architecture)
        model = builder(weights=None, weights_backbone=None, num_classes=len(self.class_names))
        if state_dict is not None:
            model.load_state_dict(state_dict)
            logger.info("Loaded Faster R-CNN weights from %s", weights_path)
        else:
            self.model_version = f"{cfg.architecture}-untrained"
            logger.warning("Faster R-CNN running with untrained heads (LOCALIZATION_ALLOW_UNTRAINED).")

        model.to(device)
        model.eval()
        self._torch, self._device, self._model = torch, device, model
        self.load_error = None
        logger.info("Faster R-CNN document localizer ready on %s", device)

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> Any:
        requested = (requested or "auto").strip().lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested for localization but unavailable; using CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    @staticmethod
    def _torch_load(torch: Any, path: Path, device: Any) -> Any:
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:  # torch < 1.13 has no weights_only argument
            return torch.load(path, map_location=device)

    def _unpack_checkpoint(self, checkpoint: Any) -> dict:
        """Accept a bare state_dict or a training checkpoint carrying metadata."""
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            class_names = checkpoint.get("class_names")
            if class_names:
                self.class_names = list(class_names)
                if self.class_names[0] != "background":
                    self.class_names.insert(0, "background")
            if checkpoint.get("model_version"):
                self.model_version = str(checkpoint["model_version"])
            return checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict):
            return checkpoint
        raise ValueError("Unrecognised checkpoint format (expected a state_dict).")

    def status(self) -> dict[str, Any]:
        """Cheap status for the health endpoint — does not force a model load."""
        cfg = self.config
        torch_available = (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("torchvision") is not None
        )
        weights_present = Path(cfg.model_path).is_file()
        if not cfg.enabled:
            active = "disabled"
        elif self.model_loaded:
            active = "faster_rcnn"
        elif torch_available and (weights_present or cfg.allow_untrained) and not self._load_attempted:
            active = "faster_rcnn (not loaded yet)"
        else:
            active = cfg.fallback_mode
        return {
            "active_backend": active,
            "torch_available": torch_available,
            "weights_present": weights_present,
            "device": self.device,
            "fallback_mode": cfg.fallback_mode,
            "load_error": self.load_error,
        }

    # -- inference -----------------------------------------------------------

    def detect_regions(
        self, image_bgr: np.ndarray, document_type: str | None = None
    ) -> RegionLocalizationResult:
        """Localize document regions. Never raises for model/runtime failures;
        those degrade to the configured fallback with a warning."""
        start = time.perf_counter()
        image = _ensure_bgr(image_bgr)
        h, w = image.shape[:2]
        warnings: list[str] = []
        cfg = self.config

        if not cfg.enabled:
            regions: list[DetectedRegion] = []
            backend, version, fallback_used = "disabled", "disabled", False
            warnings.append("Region localization is disabled; downstream modules scan the full image.")
        elif self.load():
            try:
                regions = self._detect_with_model(image)
                backend, version, fallback_used = "faster_rcnn", self.model_version, False
            except Exception as e:
                logger.exception("Faster R-CNN inference failed; using fallback.")
                warnings.append(f"Faster R-CNN inference failed ({e}); used {cfg.fallback_mode} fallback.")
                regions, backend, version = self._fallback_regions(image, document_type)
                fallback_used = True
        else:
            warnings.append(f"Faster R-CNN unavailable: {self.load_error} Using {cfg.fallback_mode} fallback.")
            regions, backend, version = self._fallback_regions(image, document_type)
            fallback_used = True

        expected = list(DOCUMENT_TYPE_REGIONS.get(document_type or "", []))
        if backend == "faster_rcnn" and expected:
            allowed = set(expected)
            regions = [r for r in regions if r.label in allowed]

        found = {r.label for r in regions}
        missing: list[str] = []
        completeness: float | None = None
        if backend == "faster_rcnn" and expected:
            missing = [label for label in expected if label not in found]
            completeness = round((len(expected) - len(missing)) / len(expected), 3)
            critical_missing = [c for c in CRITICAL_REGIONS.get(document_type or "", []) if c in missing]
            if critical_missing:
                warnings.append(f"Expected region(s) not found: {', '.join(critical_missing)}.")

        confidences = [r.bbox.confidence for r in regions]
        mean_conf = round(float(np.mean(confidences)), 4) if confidences else None

        if cfg.save_crops and regions:
            self._save_crops(image, regions, warnings)

        return RegionLocalizationResult(
            regions=regions,
            processing_time_ms=round((time.perf_counter() - start) * 1000.0, 2),
            model_version=version,
            backend=backend,
            fallback_used=fallback_used,
            image_width=int(w),
            image_height=int(h),
            expected_labels=expected,
            missing_labels=missing,
            completeness=completeness,
            mean_confidence=mean_conf,
            warnings=warnings,
        )

    def _detect_with_model(self, image_bgr: np.ndarray) -> list[DetectedRegion]:
        torch = self._torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
        with torch.inference_mode():
            prediction = self._model([tensor.to(self._device)])[0]

        return self.postprocess_predictions(
            prediction["boxes"].detach().cpu().numpy(),
            prediction["labels"].detach().cpu().numpy(),
            prediction["scores"].detach().cpu().numpy(),
            image_bgr.shape,
        )

    def postprocess_predictions(
        self,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> list[DetectedRegion]:
        """Map raw Faster R-CNN outputs (xyxy boxes, class indices, scores) to
        regions: per-class thresholds, clipping, and a per-label cap."""
        per_label: dict[str, list[DetectedRegion]] = {}
        for box, label_idx, score in zip(boxes, labels, scores):
            idx = int(label_idx)
            if idx <= 0 or idx >= len(self.class_names):
                continue
            label = self.class_names[idx]
            if float(score) < self.config.threshold_for(label):
                continue
            bbox = make_bbox(box[0], box[1], box[2], box[3], float(score), image_shape)
            if bbox.width < 2 or bbox.height < 2:
                continue
            per_label.setdefault(label, []).append(DetectedRegion(label=label, bbox=bbox))

        regions: list[DetectedRegion] = []
        for label_regions in per_label.values():
            label_regions.sort(key=lambda r: r.bbox.confidence, reverse=True)
            regions.extend(label_regions[: self.config.max_detections_per_label])
        regions.sort(key=lambda r: (r.bbox.ymin, r.bbox.xmin))
        return regions

    def _fallback_regions(
        self, image_bgr: np.ndarray, document_type: str | None
    ) -> tuple[list[DetectedRegion], str, str]:
        mode = self.config.fallback_mode
        h, w = image_bgr.shape[:2]
        if mode == "none":
            return [], "none", "none"
        if mode == "full_image":
            full = make_bbox(0, 0, w, h, 0.05, image_bgr.shape)
            return [DetectedRegion(label="document", bbox=full)], "full_image", FULL_IMAGE_MODEL_VERSION
        try:
            return self._heuristic_regions(image_bgr, document_type), "heuristic", HEURISTIC_MODEL_VERSION
        except Exception:
            logger.exception("Heuristic localization failed; using full-image box.")
            full = make_bbox(0, 0, w, h, 0.05, image_bgr.shape)
            return [DetectedRegion(label="document", bbox=full)], "full_image", FULL_IMAGE_MODEL_VERSION

    def _heuristic_regions(self, image_bgr: np.ndarray, document_type: str | None) -> list[DetectedRegion]:
        normalized = _contrast_normalize(image_bgr)
        doc_box = _find_document_box(normalized)
        regions = [DetectedRegion(label="document", bbox=doc_box)]

        mrz_box = _find_mrz_band(normalized, doc_box) if document_type in _MRZ_DOCUMENT_TYPES else None
        if mrz_box is not None:
            regions.append(DetectedRegion(label="mrz", bbox=mrz_box))

        photo_box = _find_photo_region(image_bgr, doc_box)
        if photo_box is not None:
            regions.append(DetectedRegion(label="photo", bbox=photo_box))

        visual_bottom = mrz_box.ymin if mrz_box is not None else doc_box.ymax
        if visual_bottom - doc_box.ymin > 10:
            regions.append(DetectedRegion(
                label="visual_zone",
                bbox=make_bbox(doc_box.xmin, doc_box.ymin, doc_box.xmax, visual_bottom,
                               min(doc_box.confidence, 0.2), image_bgr.shape),
            ))
        return regions

    def _save_crops(self, image_bgr: np.ndarray, regions: list[DetectedRegion], warnings: list[str]) -> None:
        run_dir = Path(self.config.crops_dir) / uuid.uuid4().hex
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            counts: dict[str, int] = {}
            for region in regions:
                crop = crop_region(image_bgr, region.bbox)
                if crop.size == 0:
                    continue
                idx = counts.get(region.label, 0)
                counts[region.label] = idx + 1
                path = run_dir / f"{region.label}_{idx}.png"
                if cv2.imwrite(str(path), crop):
                    try:
                        region.cropped_image_path = path.relative_to(BACKEND_ROOT).as_posix()
                    except ValueError:
                        region.cropped_image_path = path.as_posix()
        except OSError as e:
            warnings.append(f"Could not save region crops: {e}")


# ── Module-level singleton ─────────────────────────────────────────────

_detector: DocumentRegionDetector | None = None
_detector_lock = threading.Lock()


def get_detector() -> DocumentRegionDetector:
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = DocumentRegionDetector()
    return _detector


def reset_detector() -> None:
    """Drop the cached detector (used by tests after changing configuration)."""
    global _detector
    with _detector_lock:
        _detector = None


def detect_document_regions(
    image_bgr: np.ndarray, document_type: str | None = None
) -> RegionLocalizationResult:
    return get_detector().detect_regions(image_bgr, document_type=document_type)


def trusted_regions(
    result: RegionLocalizationResult | None, min_confidence: float | None = None
) -> list[DetectedRegion]:
    """Semantic regions confident enough to steer downstream modules.
    Layout-only fallback labels ("document", "visual_zone") are excluded."""
    if result is None:
        return []
    threshold = get_localization_config().min_region_confidence if min_confidence is None else min_confidence
    return [
        r for r in result.regions
        if r.label not in FALLBACK_LABELS and r.bbox.confidence >= threshold
    ]
