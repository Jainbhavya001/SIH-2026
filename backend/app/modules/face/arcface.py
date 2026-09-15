"""
ArcFace embedding engine — high-quality 512-d face embeddings via insightface.

Backends, tried in order:

1. **insightface** (buffalo_l) — RetinaFace detection + ArcFace recognition
   in one pass. Downloads ~300 MB of models on first use, then cached in
   ``~/.insightface/models/``.  ``pip install insightface onnxruntime``
2. Unavailable — callers fall through to the next verification tier
   (dlib ``face_recognition``, then histogram+ORB).

The module exposes a lazy-initialised singleton so the model loads once per
process and every subsequent call is a fast forward pass.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_engine: Any | None = None
_lock = threading.Lock()
_init_attempted = False
_init_error: str | None = None


def _get_engine() -> Any | None:
    global _engine, _init_attempted, _init_error
    if _init_attempted:
        return _engine
    with _lock:
        if _init_attempted:
            return _engine
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _engine = app
            _init_error = None
            logger.info("ArcFace engine ready (insightface buffalo_l, CPU).")
        except ImportError:
            _init_error = "insightface not installed"
            logger.info("insightface not installed; ArcFace unavailable.")
        except Exception as exc:
            _init_error = str(exc)
            logger.warning("ArcFace init failed: %s", exc)
        finally:
            _init_attempted = True
    return _engine


def is_available() -> bool:
    return _get_engine() is not None


def status() -> dict[str, Any]:
    return {
        "available": is_available(),
        "error": _init_error,
        "backend": "insightface/buffalo_l" if is_available() else None,
    }


def get_embedding(face_bgr: np.ndarray) -> np.ndarray | None:
    """Return a 512-d L2-normalised ArcFace embedding for the largest face
    visible in *face_bgr*.  The image can be a tight crop (padding is added
    automatically) or a full document / selfie frame.

    Returns ``None`` when the engine is unavailable or no face is detected.
    """
    engine = _get_engine()
    if engine is None:
        return None
    if face_bgr is None or face_bgr.size == 0:
        return None

    image = _prepare(face_bgr)
    faces = engine.get(image)
    if not faces:
        return None

    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    emb = getattr(largest, "normed_embedding", None)
    if emb is None:
        emb = getattr(largest, "embedding", None)
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32)


def get_all_embeddings(image_bgr: np.ndarray) -> list[dict]:
    """Detect every face in *image_bgr* and return a list of dicts:
    ``{"bbox": [x1,y1,x2,y2], "embedding": np.ndarray(512,), "det_score": float}``
    """
    engine = _get_engine()
    if engine is None or image_bgr is None or image_bgr.size == 0:
        return []

    image = _prepare(image_bgr)
    faces = engine.get(image)
    results = []
    for f in faces:
        emb = getattr(f, "normed_embedding", None) or getattr(f, "embedding", None)
        if emb is None:
            continue
        results.append({
            "bbox": [float(v) for v in f.bbox],
            "embedding": np.asarray(emb, dtype=np.float32),
            "det_score": float(f.det_score) if hasattr(f, "det_score") else 0.0,
        })
    return results


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1], clamped to [0, 1] for the API."""
    na, nb = np.linalg.norm(emb_a), np.linalg.norm(emb_b)
    if na == 0 or nb == 0:
        return 0.0
    return float(max(0.0, np.dot(emb_a, emb_b) / (na * nb)))


# ── helpers ────────────────────────────────────────────────────────────

def _prepare(image_bgr: np.ndarray) -> np.ndarray:
    """Ensure a BGR uint8 image with enough padding for insightface's
    RetinaFace detector to latch onto a face in a tight crop."""
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

    h, w = image_bgr.shape[:2]
    short_side = min(h, w)
    if short_side < 80:
        scale = 160.0 / short_side
        image_bgr = cv2.resize(
            image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
        )
        h, w = image_bgr.shape[:2]

    pad = int(max(h, w) * 0.25)
    if pad > 0:
        image_bgr = cv2.copyMakeBorder(
            image_bgr, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=(128, 128, 128),
        )
    return image_bgr
