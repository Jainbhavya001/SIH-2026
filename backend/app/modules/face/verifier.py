"""
Module 4 — Face Verification.

Compares the face on the document against a live-captured photo of the
person presenting it, and returns a similarity score.

Two backends, auto-selected at import time:

1. **`face_recognition` (dlib embeddings)** — if installed, this is used.
   It produces a 128-d face embedding and a Euclidean distance is a very
   reliable same-person/different-person signal. Not installed by default
   because `dlib` needs CMake + a C++ toolchain, which is a heavy ask on a
   judge's/teammate's Windows machine mid-hackathon.
2. **Lightweight fallback (default)** — aligns both faces to a fixed size
   and combines (a) grayscale histogram correlation and (b) ORB descriptor
   match ratio. This is meaningfully weaker than a learned embedding but
   needs zero extra native dependencies, so the demo always runs. The
   README documents exactly how to upgrade to backend 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import get_settings

try:
    import face_recognition  # type: ignore

    _STRONG_BACKEND = True
except ImportError:
    _STRONG_BACKEND = False

_ALIGN_SIZE = (200, 200)


@dataclass
class FaceMatchResult:
    similarity: float  # 0..1, higher = more likely same person
    is_match: bool
    backend: str
    document_face_found: bool
    live_face_found: bool


def _prep(face_img: np.ndarray) -> np.ndarray:
    resized = cv2.resize(face_img, _ALIGN_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    return cv2.equalizeHist(gray)


def _fallback_similarity(face_a: np.ndarray, face_b: np.ndarray) -> float:
    gray_a, gray_b = _prep(face_a), _prep(face_b)

    hist_a = cv2.calcHist([gray_a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [256], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    hist_score = max(0.0, cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))

    orb = cv2.ORB_create(nfeatures=500)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    orb_score = 0.0
    if des_a is not None and des_b is not None and len(kp_a) > 5 and len(kp_b) > 5:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_a, des_b)
        good = [m for m in matches if m.distance < 50]
        orb_score = min(1.0, len(good) / max(1, min(len(kp_a), len(kp_b))))

    # Weighted blend: histogram correlation captures overall tone/shape
    # similarity, ORB match ratio captures structural (feature-point) similarity.
    return round(0.5 * hist_score + 0.5 * orb_score, 4)


def _strong_similarity(face_a: np.ndarray, face_b: np.ndarray) -> float | None:
    rgb_a = cv2.cvtColor(face_a, cv2.COLOR_BGR2RGB)
    rgb_b = cv2.cvtColor(face_b, cv2.COLOR_BGR2RGB)

    enc_a = face_recognition.face_encodings(rgb_a)
    enc_b = face_recognition.face_encodings(rgb_b)
    if not enc_a or not enc_b:
        return None

    distance = float(np.linalg.norm(enc_a[0] - enc_b[0]))
    # face_recognition's typical match threshold is distance < 0.6; convert
    # to a 0..1 similarity for a consistent API with the fallback backend.
    similarity = max(0.0, 1.0 - distance / 1.2)
    return round(similarity, 4)


def compare_faces(document_face: np.ndarray | None, live_face: np.ndarray | None) -> FaceMatchResult:
    settings = get_settings()

    if document_face is None or live_face is None:
        return FaceMatchResult(
            similarity=0.0,
            is_match=False,
            backend="none",
            document_face_found=document_face is not None,
            live_face_found=live_face is not None,
        )

    if _STRONG_BACKEND:
        similarity = _strong_similarity(document_face, live_face)
        if similarity is not None:
            return FaceMatchResult(
                similarity=similarity,
                is_match=similarity >= settings.FACE_MATCH_THRESHOLD,
                backend="face_recognition (dlib embeddings)",
                document_face_found=True,
                live_face_found=True,
            )
        # fall through to lightweight backend if dlib couldn't encode either face

    similarity = _fallback_similarity(document_face, live_face)
    return FaceMatchResult(
        similarity=similarity,
        is_match=similarity >= settings.FACE_MATCH_THRESHOLD,
        backend="histogram+ORB (lightweight fallback)",
        document_face_found=True,
        live_face_found=True,
    )
