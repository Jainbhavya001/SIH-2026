"""
Face detection with automatic backend selection.

Backends (tried in order):
  1. ``cv2.FaceDetectorYN``  — cv2 5.0+, needs an ONNX model file
     (``face_detection_yunet_2023mar.onnx``, auto-downloaded on first use).
  2. ``cv2.CascadeClassifier`` — cv2 < 5.0, bundled Haar cascade.
  3. insightface RetinaFace — if the ArcFace module loaded successfully.
  4. No detection (returns empty list).

When Module 0 (region localization) has found the document's portrait
photo, the search is restricted to that region first: it removes false
positives from printed patterns/holograms and lets small portraits be
upscaled past the detector's minimum face size. The full-image scan
remains the fallback.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

from app.models.schemas import BoundingBox

logger = logging.getLogger(__name__)

_REGION_MIN_SIDE = 240
_REGION_PADDING_RATIO = 0.15

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_YUNET_MODEL = _MODELS_DIR / "face_detection_yunet_2023mar.onnx"
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


# ── Backend initialisation ──────────────────────────────────────────────

_backend: str = "none"
_face_cascade = None
_yunet_lock = threading.Lock()
_yunet_downloaded = False

# Try Haar cascade (cv2 < 5.0)
if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
    try:
        _cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.isfile(_cascade_path):
            _face_cascade = cv2.CascadeClassifier(_cascade_path)
            if not _face_cascade.empty():
                _backend = "haar"
    except Exception:
        _face_cascade = None


def _ensure_yunet_model() -> bool:
    global _yunet_downloaded
    if _yunet_downloaded:
        return _YUNET_MODEL.is_file()
    with _yunet_lock:
        if _YUNET_MODEL.is_file():
            _yunet_downloaded = True
            return True
        try:
            _MODELS_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading YuNet face detection model ...")
            urlretrieve(_YUNET_URL, str(_YUNET_MODEL))
            _yunet_downloaded = True
            logger.info("YuNet model saved to %s", _YUNET_MODEL)
            return True
        except Exception:
            logger.warning("Could not download YuNet model; face detection degraded.", exc_info=True)
            return False


def _detect_yunet(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = gray.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(_YUNET_MODEL), "", (w, h))
    detector.setScoreThreshold(0.7)
    _, faces = detector.detect(gray)
    if faces is None:
        return []
    results = []
    for face in faces:
        fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        if fw > 0 and fh > 0:
            results.append((fx, fy, fw, fh))
    return results


def _detect_haar(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    if _face_cascade is None or _face_cascade.empty():
        return []
    eq = cv2.equalizeHist(gray)
    detections = _face_cascade.detectMultiScale(eq, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detections]


def _detect_insightface(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    try:
        from app.modules.face.arcface import _get_engine
        engine = _get_engine()
        if engine is None:
            return []
        faces = engine.get(image_bgr)
        results = []
        for f in faces:
            x0, y0, x1, y1 = f.bbox.astype(int)
            w, h = x1 - x0, y1 - y0
            if w > 0 and h > 0:
                results.append((int(x0), int(y0), int(w), int(h)))
        return results
    except Exception:
        return []


# ── Public API ──────────────────────────────────────────────────────────

@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    source: str = "full_image"


def detect_faces(image: np.ndarray) -> list[FaceBox]:
    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # 1. Try YuNet (cv2 5.0+)
    if hasattr(cv2, "FaceDetectorYN") and _ensure_yunet_model():
        raw = _detect_yunet(gray)
        if raw:
            return [FaceBox(x, y, w, h) for (x, y, w, h) in raw]

    # 2. Try Haar cascade (cv2 < 5.0)
    if _backend == "haar":
        raw = _detect_haar(gray)
        if raw:
            return [FaceBox(x, y, w, h) for (x, y, w, h) in raw]

    # 3. Try insightface RetinaFace
    if image.ndim == 3:
        raw = _detect_insightface(image)
        if raw:
            return [FaceBox(x, y, w, h) for (x, y, w, h) in raw]

    return []


def detect_faces_in_region(image: np.ndarray, region: BoundingBox) -> list[FaceBox]:
    """Detect faces inside a (padded) region; boxes in full-image coordinates."""
    h, w = image.shape[:2]
    pad_x = region.width * _REGION_PADDING_RATIO
    pad_y = region.height * _REGION_PADDING_RATIO
    x0 = int(max(0, np.floor(region.xmin - pad_x)))
    y0 = int(max(0, np.floor(region.ymin - pad_y)))
    x1 = int(min(w, np.ceil(region.xmax + pad_x)))
    y1 = int(min(h, np.ceil(region.ymax + pad_y)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return []

    crop = image[y0:y1, x0:x1]
    scale = 1.0
    min_side = min(crop.shape[:2])
    if min_side < _REGION_MIN_SIDE:
        scale = _REGION_MIN_SIDE / float(min_side)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    faces: list[FaceBox] = []
    for f in detect_faces(crop):
        fx, fy = x0 + int(round(f.x / scale)), y0 + int(round(f.y / scale))
        fw, fh = int(round(f.w / scale)), int(round(f.h / scale))
        fw, fh = min(fw, w - fx), min(fh, h - fy)
        if fw > 0 and fh > 0:
            faces.append(FaceBox(fx, fy, fw, fh, source="photo_region"))
    return faces


def detect_largest_face(
    image: np.ndarray, photo_region: BoundingBox | None = None
) -> tuple[np.ndarray, FaceBox] | None:
    faces: list[FaceBox] = []
    if photo_region is not None:
        faces = detect_faces_in_region(image, photo_region)
    if not faces:
        faces = detect_faces(image)
    if not faces:
        return None
    largest = max(faces, key=lambda f: f.w * f.h)
    crop = image[largest.y : largest.y + largest.h, largest.x : largest.x + largest.w]
    return crop, largest
