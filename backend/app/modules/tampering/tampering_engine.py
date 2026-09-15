"""
Module 3 — Tampering Detection (core AI innovation).

Aggregates ELA, metadata forensics, and copy-move detection into one
sub-score (0 = clean, 100 = heavily tampered) with an evidence list. Each
sub-signal is weighted by how reliable it tends to be in isolation: ELA is
the strongest single signal for photo/text replacement, copy-move is a
strong but narrower signal (only fires on literal clone-stamping), and
metadata is corroborating evidence only.

When Module 0 has localized the document's regions, high-risk regions
(photo, name, dates, document number, MRZ) are also inspected one by one.
A photo swap or an edited date can be too small to move whole-image
statistics, but stands out when that region is compared with the rest of
the document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from app.models.schemas import DetectedRegion
from app.modules.localization.config import HIGH_RISK_LABELS, get_localization_config
from app.modules.localization.detector import bbox_to_xywh
from app.modules.tampering.copy_move import CopyMoveResult, detect_copy_move
from app.modules.tampering.ela import ELAResult, run_ela
from app.modules.tampering.metadata_analysis import MetadataResult, analyze_metadata

_ELA_WEIGHT = 0.5
_COPY_MOVE_WEIGHT = 0.35
_METADATA_WEIGHT = 0.15

# Region-level thresholds. ELA error naturally rises with texture (a portrait
# or dense text has more edges than a blank background), so a region is only
# flagged when its error is well above what the *rest of the document*
# shows for the same amount of texture.
_REGION_MIN_PIXELS = 64
_BLOCK_SIZE = 16
_TEXTURE_BINS = 8
_REGION_MIN_BASELINE_BLOCKS = 16
_REGION_MIN_HOT_RATIO = 0.05
_REGION_HOT_RATIO_MULTIPLIER = 3.0
_REGION_TEXTURE_RATIO_THRESHOLD = 2.0
_REGION_COPY_MOVE_MIN_HITS = 6
# Printed text repeats identical glyphs ("<<<<" in the MRZ, repeated letters),
# which ORB self-matching cannot tell apart from cloning, so copy-move hits
# are only attributed to pictorial regions.
_COPY_MOVE_REGION_LABELS = frozenset({"photo"})
_REGION_POINTS_PER_FINDING = 20.0
_REGION_COMPONENT_CAP = 30.0


@dataclass
class RegionTamperFinding:
    label: str
    ela_hot_ratio: float
    ela_mean_error: float
    copy_move_hits: int
    suspicious: bool
    region_confidence: float
    reason: str | None = None


@dataclass
class TamperingResult:
    tampering_score: int  # 0 (clean) - 100 (heavily suspicious)
    verdict: str  # "clean" | "suspicious" | "tampered"
    evidence: list[str] = field(default_factory=list)
    ela: ELAResult | None = None
    copy_move: CopyMoveResult | None = None
    metadata: MetadataResult | None = None
    region_findings: list[RegionTamperFinding] = field(default_factory=list)


def _texture_map(image_array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if image_array.ndim == 3 else image_array
    if gray.shape[:2] != shape:
        gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _expected_error_for_texture(
    error_blocks: np.ndarray,
    texture_blocks: np.ndarray,
    exclude: tuple[int, int, int, int],
    region_texture: float,
) -> float | None:
    """
    Expected ELA error for a region with `region_texture`, learnt from the
    rest of the document: blocks are grouped into texture quantile bins and
    the median error per bin is interpolated. The region's own blocks are
    excluded so an edited region cannot raise its own baseline, and medians
    keep other edited areas from skewing it.
    """
    by0, by1, bx0, bx1 = exclude
    mask = np.ones(error_blocks.shape, dtype=bool)
    mask[by0:by1, bx0:bx1] = False
    errors, textures = error_blocks[mask], texture_blocks[mask]
    if errors.size < _REGION_MIN_BASELINE_BLOCKS:
        return None

    edges = np.unique(np.quantile(textures, np.linspace(0.0, 1.0, _TEXTURE_BINS + 1)))
    if len(edges) < 2:
        return float(np.median(errors))
    bins = np.clip(np.searchsorted(edges, textures, side="right") - 1, 0, len(edges) - 2)
    centres: list[float] = []
    medians: list[float] = []
    for b in range(len(edges) - 1):
        selected = bins == b
        if selected.any():
            centres.append(float(np.median(textures[selected])))
            medians.append(float(np.median(errors[selected])))
    order = np.argsort(centres)
    return float(np.interp(region_texture, np.asarray(centres)[order], np.asarray(medians)[order]))


def analyze_regions(
    image_array: np.ndarray,
    ela_result: ELAResult,
    copy_move_result: CopyMoveResult,
    regions: Sequence[DetectedRegion],
) -> list[RegionTamperFinding]:
    """Inspect each confidently-localized high-risk region for ELA and copy-move anomalies."""
    min_conf = get_localization_config().min_region_confidence
    candidates = [r for r in regions if r.label in HIGH_RISK_LABELS and r.bbox.confidence >= min_conf]
    if not candidates:
        return []

    error_map = ela_result.error_map
    img_h, img_w = image_array.shape[:2]
    ela_usable = error_map is not None and error_map.size > 0
    if ela_usable:
        map_h, map_w = error_map.shape[:2]
        sx, sy = map_w / float(img_w), map_h / float(img_h)
        # A transposed map (EXIF rotation applied by one decoder but not the
        # other) cannot be aligned with the boxes, so skip ELA for regions.
        ela_usable = abs(sx - sy) <= 0.02
    if ela_usable:
        texture = _texture_map(image_array, (map_h, map_w))
        blocks_h, blocks_w = max(1, map_h // _BLOCK_SIZE), max(1, map_w // _BLOCK_SIZE)
        error_blocks = cv2.resize(error_map.astype(np.float32), (blocks_w, blocks_h), interpolation=cv2.INTER_AREA)
        texture_blocks = cv2.resize(texture, (blocks_w, blocks_h), interpolation=cv2.INTER_AREA)
        block_sy, block_sx = blocks_h / float(map_h), blocks_w / float(map_w)
        hot_ratio_floor = max(_REGION_MIN_HOT_RATIO, ela_result.suspicious_region_ratio * _REGION_HOT_RATIO_MULTIPLIER)

    points = np.asarray(copy_move_result.matched_points, dtype=np.float32).reshape(-1, 2)

    findings: list[RegionTamperFinding] = []
    for region in candidates:
        x, y, w, h = bbox_to_xywh(region.bbox)
        reasons: list[str] = []
        hot_ratio = mean_error = 0.0

        if ela_usable:
            mx0, my0 = int(x * sx), int(y * sy)
            mx1, my1 = int(np.ceil((x + w) * sx)), int(np.ceil((y + h) * sy))
            patch = error_map[my0:my1, mx0:mx1]
            if patch.size >= _REGION_MIN_PIXELS:
                hot_ratio = float((patch > ela_result.hot_threshold).mean())
                mean_error = float(patch.mean())
                exclude = (
                    int(my0 * block_sy), int(np.ceil(my1 * block_sy)),
                    int(mx0 * block_sx), int(np.ceil(mx1 * block_sx)),
                )
                expected = _expected_error_for_texture(
                    error_blocks, texture_blocks, exclude, float(texture[my0:my1, mx0:mx1].mean())
                )
                relative = mean_error / (expected + 1.0) if expected is not None else 0.0
                if hot_ratio >= hot_ratio_floor and relative >= _REGION_TEXTURE_RATIO_THRESHOLD:
                    reasons.append(
                        f"compression error {relative:.1f}x the document baseline for its texture "
                        f"({hot_ratio * 100:.1f}% hot pixels)"
                    )

        hits = 0
        if len(points):
            inside = (
                (points[:, 0] >= x) & (points[:, 0] <= x + w)
                & (points[:, 1] >= y) & (points[:, 1] <= y + h)
            )
            hits = int(inside.sum())
            if (
                region.label in _COPY_MOVE_REGION_LABELS
                and copy_move_result.suspicious
                and hits >= _REGION_COPY_MOVE_MIN_HITS
            ):
                reasons.append(f"{hits} cloned-keypoint matches fall inside this region")

        findings.append(RegionTamperFinding(
            label=region.label,
            ela_hot_ratio=round(hot_ratio, 4),
            ela_mean_error=round(mean_error, 3),
            copy_move_hits=hits,
            suspicious=bool(reasons),
            region_confidence=region.bbox.confidence,
            reason="; ".join(reasons) if reasons else None,
        ))
    return findings


def analyze_tampering(
    image_bytes: bytes,
    image_array: np.ndarray,
    regions: Sequence[DetectedRegion] | None = None,
) -> TamperingResult:
    evidence: list[str] = []

    ela_result = run_ela(image_bytes, return_error_map=bool(regions))
    ela_component = min(100, ela_result.suspicious_region_ratio * 2500)
    if ela_result.suspicious:
        evidence.append(
            f"Error Level Analysis found {ela_result.suspicious_region_ratio * 100:.1f}% of the image "
            f"with abnormal compression error (max error {ela_result.max_error:.0f}) — consistent with "
            f"a pasted or re-edited region (photo swap, altered text, or added stamp)."
        )

    copy_move_result = detect_copy_move(image_array)
    copy_move_component = min(100, copy_move_result.match_count * 4)
    if copy_move_result.suspicious:
        evidence.append(
            f"Detected {copy_move_result.match_count} self-similar keypoint matches between distant "
            f"regions of the image — consistent with copy-move (clone-stamp) forgery, e.g. a duplicated "
            f"stamp or hologram patch."
        )

    metadata_result = analyze_metadata(image_bytes)
    metadata_component = 0.0
    if metadata_result.editor_signature_found:
        metadata_component = 60.0
        evidence.append(f"Image metadata shows it was processed with '{metadata_result.software_tag}'.")
    elif metadata_result.flags and metadata_result.has_exif:
        metadata_component = 20.0
    for f in metadata_result.flags:
        if f not in evidence:
            evidence.append(f)

    region_findings: list[RegionTamperFinding] = []
    region_component = 0.0
    if regions:
        region_findings = analyze_regions(image_array, ela_result, copy_move_result, regions)
        for finding in region_findings:
            if not finding.suspicious:
                continue
            region_component += _REGION_POINTS_PER_FINDING * finding.region_confidence
            evidence.append(
                f"Localized '{finding.label}' region looks altered: {finding.reason} — consistent with "
                f"a targeted edit such as a photo swap or changed identity data."
            )
        region_component = min(_REGION_COMPONENT_CAP, region_component)
        ela_result.error_map = None  # large array; not needed past this point

    score = (
        ela_component * _ELA_WEIGHT
        + copy_move_component * _COPY_MOVE_WEIGHT
        + metadata_component * _METADATA_WEIGHT
        + region_component
    )
    score = int(round(min(100, score)))

    if score < 20:
        verdict = "clean"
    elif score < 55:
        verdict = "suspicious"
    else:
        verdict = "tampered"

    if not evidence:
        evidence.append("No tampering indicators detected by ELA, copy-move analysis, or metadata inspection.")

    return TamperingResult(
        tampering_score=score,
        verdict=verdict,
        evidence=evidence,
        ela=ela_result,
        copy_move=copy_move_result,
        metadata=metadata_result,
        region_findings=region_findings,
    )
