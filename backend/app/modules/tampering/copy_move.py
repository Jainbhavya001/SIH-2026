"""
Copy-move forgery detection via ORB keypoint self-matching.

Classic "copy-move" forgery: someone clones a region of the *same* image
(a clean stamp, a hologram patch, a signature) and pastes it elsewhere to
cover an alteration. Because the cloned region is pixel-identical (or
near-identical after light blending) to somewhere else in the image, we
can detect it by matching the image's ORB keypoints against *itself* and
looking for clusters of strong matches between two distant regions.

This runs in milliseconds with OpenCV alone — no external models needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import get_settings

_MIN_KEYPOINT_DISTANCE = 24  # px; matches closer than this are just noise/texture repetition
_MAX_REPORTED_PAIRS = 1000


@dataclass
class CopyMoveResult:
    match_count: int
    suspicious: bool
    matched_regions: list[tuple[int, int, int, int]] = field(default_factory=list)  # x,y,w,h boxes
    # Keypoint locations taking part in strong matches (capped), used to
    # attribute copy-move evidence to localized document regions.
    matched_points: list[tuple[float, float]] = field(default_factory=list)


def detect_copy_move(image: np.ndarray) -> CopyMoveResult:
    settings = get_settings()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    orb = cv2.ORB_create(nfeatures=1500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 10:
        return CopyMoveResult(match_count=0, suspicious=False)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, descriptors, k=3)

    strong_pairs: list[tuple[int, int]] = []
    for match_group in matches:
        # k=3 because the top match of a descriptor against itself is always
        # itself (distance 0) — skip it and look at the next-best matches.
        for m in match_group[1:]:
            if m.distance < 35:
                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[m.trainIdx].pt
                dist = float(np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]))
                if dist > _MIN_KEYPOINT_DISTANCE:
                    strong_pairs.append((m.queryIdx, m.trainIdx))

    match_count = len(strong_pairs)
    suspicious = match_count >= settings.COPY_MOVE_MATCH_THRESHOLD

    matched_points = [keypoints[i].pt for pair in strong_pairs[:_MAX_REPORTED_PAIRS] for i in pair]

    regions: list[tuple[int, int, int, int]] = []
    if suspicious:
        pts = np.array([keypoints[i].pt for pair in strong_pairs for i in pair])
        if len(pts):
            x, y = pts[:, 0], pts[:, 1]
            regions.append((int(x.min()), int(y.min()), int(x.max() - x.min()), int(y.max() - y.min())))

    return CopyMoveResult(
        match_count=match_count,
        suspicious=suspicious,
        matched_regions=regions,
        matched_points=[(float(x), float(y)) for x, y in matched_points],
    )
