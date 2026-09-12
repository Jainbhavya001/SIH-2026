"""
Module 3 — Tampering Detection (core AI innovation).

Aggregates ELA, metadata forensics, and copy-move detection into one
sub-score (0 = clean, 100 = heavily tampered) with an evidence list. Each
sub-signal is weighted by how reliable it tends to be in isolation: ELA is
the strongest single signal for photo/text replacement, copy-move is a
strong but narrower signal (only fires on literal clone-stamping), and
metadata is corroborating evidence only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.modules.tampering.copy_move import CopyMoveResult, detect_copy_move
from app.modules.tampering.ela import ELAResult, run_ela
from app.modules.tampering.metadata_analysis import MetadataResult, analyze_metadata

_ELA_WEIGHT = 0.5
_COPY_MOVE_WEIGHT = 0.35
_METADATA_WEIGHT = 0.15


@dataclass
class TamperingResult:
    tampering_score: int  # 0 (clean) - 100 (heavily suspicious)
    verdict: str  # "clean" | "suspicious" | "tampered"
    evidence: list[str] = field(default_factory=list)
    ela: ELAResult | None = None
    copy_move: CopyMoveResult | None = None
    metadata: MetadataResult | None = None


def analyze_tampering(image_bytes: bytes, image_array: np.ndarray) -> TamperingResult:
    evidence: list[str] = []

    ela_result = run_ela(image_bytes)
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

    score = (
        ela_component * _ELA_WEIGHT
        + copy_move_component * _COPY_MOVE_WEIGHT
        + metadata_component * _METADATA_WEIGHT
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
    )
