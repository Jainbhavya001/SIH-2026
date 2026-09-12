"""
Face detection using OpenCV's bundled Haar cascade (zero extra downloads,
works out of the box on every platform including Windows without native
build tools). If a DNN face-detector model file is dropped into
`backend/app/modules/face/models/`, swap it in here for better accuracy —
the rest of the pipeline only depends on `detect_largest_face`'s return
shape, not on which detector produced it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int


def detect_faces(image: np.ndarray) -> list[FaceBox]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.equalizeHist(gray)
    detections = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detections]


def detect_largest_face(image: np.ndarray) -> tuple[np.ndarray, FaceBox] | None:
    """Returns the cropped face image and its bounding box, or None if no face found."""
    faces = detect_faces(image)
    if not faces:
        return None
    largest = max(faces, key=lambda f: f.w * f.h)
    crop = image[largest.y : largest.y + largest.h, largest.x : largest.x + largest.w]
    return crop, largest
