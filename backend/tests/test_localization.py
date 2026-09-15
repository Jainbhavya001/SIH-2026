"""
Unit tests for Module 0 (document region localization) and the
region-aware behaviour it enables downstream. Everything here runs on a
CPU-only machine without torch or trained weights; the single real-model
test is skipped unless torch/torchvision are installed.
"""
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import get_settings
from app.models.schemas import BoundingBox, DetectedRegion, RegionLocalizationResult
from app.modules.face import detector as face_detector
from app.modules.localization.config import LocalizationConfig, get_localization_config
from app.modules.localization.dataset import DocumentRegionDataset, collate_fn, write_synthetic_coco
from app.modules.localization.detector import (
    DocumentRegionDetector,
    bbox_to_xywh,
    crop_region,
    make_bbox,
    trusted_regions,
)
from app.modules.localization.synthetic_generator import SyntheticDocumentGenerator
from app.modules.ocr.field_extractors import merge_region_fields, normalize_region_date, region_field_values
from app.modules.ocr.mrz_parser import parse_mrz
from app.modules.risk.risk_engine import compute_risk
from app.modules.tampering.copy_move import CopyMoveResult
from app.modules.tampering.ela import ELAResult
from app.modules.tampering.tampering_engine import TamperingResult, analyze_regions, analyze_tampering
from app.modules.validation.rules_engine import ValidationResult, validate_passport

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
VALID_LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def _config(tmp_path: Path, **overrides) -> LocalizationConfig:
    """A config guaranteed to have no weights, so the fallback path is exercised
    whether or not torch is installed on the machine running the tests."""
    base = LocalizationConfig(model_path=tmp_path / "missing.pth", crops_dir=tmp_path / "crops")
    return replace(base, **overrides)


@pytest.fixture(scope="module")
def synthetic_passport():
    return SyntheticDocumentGenerator().generate("passport", seed=1)


# ── Configuration ──────────────────────────────────────────────────────

def test_config_defaults_and_class_thresholds():
    cfg = LocalizationConfig()
    assert cfg.class_names[0] == "background"
    assert "mrz" in cfg.class_names and "photo" in cfg.class_names
    assert cfg.fallback_mode == "heuristic"
    assert cfg.allow_untrained is False
    assert cfg.threshold_for("mrz") == 0.5
    assert cfg.threshold_for("unknown_label") == cfg.confidence_threshold


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LOCALIZATION_FALLBACK", "full_image")
    monkeypatch.setenv("LOCALIZATION_DEVICE", "cpu")
    monkeypatch.setenv("LOCALIZATION_ARCHITECTURE", "not_a_real_arch")
    get_localization_config.cache_clear()
    try:
        cfg = get_localization_config()
        assert cfg.fallback_mode == "full_image"
        assert cfg.device == "cpu"
        assert cfg.architecture == "fasterrcnn_resnet50_fpn"
    finally:
        get_localization_config.cache_clear()


# ── Detector initialisation and fallbacks ─────────────────────────────

def test_detector_initialises_lazily(tmp_path):
    detector = DocumentRegionDetector(_config(tmp_path))
    assert detector.model_loaded is False
    status = detector.status()
    assert status["active_backend"] == "heuristic"
    assert status["weights_present"] is False


def test_missing_weights_uses_heuristic_fallback(tmp_path, synthetic_passport):
    detector = DocumentRegionDetector(_config(tmp_path))
    result = detector.detect_regions(synthetic_passport.image, document_type="passport")

    assert detector.load() is False
    assert detector.load_error
    assert result.backend == "heuristic"
    assert result.fallback_used is True
    assert result.model_version == "heuristic-opencv-v1"
    assert any("Faster R-CNN unavailable" in w for w in result.warnings)
    assert result.image_width == 900 and result.image_height == 600
    assert result.completeness is None  # fallbacks never claim completeness
    assert any(r.label == "document" for r in result.regions)


def test_heuristic_localizes_mrz_band(tmp_path, synthetic_passport):
    result = DocumentRegionDetector(_config(tmp_path)).detect_regions(synthetic_passport.image, "passport")
    mrz = result.best_region("mrz")
    assert mrz is not None
    gt_x, gt_y, gt_w, gt_h = next(a.bbox for a in synthetic_passport.annotations if a.class_name == "mrz")
    centre_y = (mrz.bbox.ymin + mrz.bbox.ymax) / 2
    assert gt_y <= centre_y <= gt_y + gt_h
    assert mrz.bbox.xmin <= gt_x + 20
    assert 0 < mrz.bbox.confidence < 0.5  # heuristic boxes stay low-confidence


def test_heuristic_skips_mrz_for_documents_without_one(tmp_path, synthetic_passport):
    result = DocumentRegionDetector(_config(tmp_path)).detect_regions(synthetic_passport.image, "driving_license")
    assert result.best_region("mrz") is None


def test_full_image_fallback(tmp_path):
    image = np.full((120, 200, 3), 255, dtype=np.uint8)
    result = DocumentRegionDetector(_config(tmp_path, fallback_mode="full_image")).detect_regions(image)
    assert result.backend == "full_image"
    assert len(result.regions) == 1
    box = result.regions[0].bbox
    assert (box.xmin, box.ymin, box.xmax, box.ymax) == (0.0, 0.0, 200.0, 120.0)
    assert box.confidence < 0.1


def test_none_fallback_and_disabled(tmp_path):
    image = np.full((50, 50, 3), 255, dtype=np.uint8)
    assert DocumentRegionDetector(_config(tmp_path, fallback_mode="none")).detect_regions(image).regions == []
    disabled = DocumentRegionDetector(_config(tmp_path, enabled=False)).detect_regions(image)
    assert disabled.backend == "disabled"
    assert disabled.regions == []


def test_inference_failure_degrades_to_fallback(tmp_path, monkeypatch):
    detector = DocumentRegionDetector(_config(tmp_path, fallback_mode="full_image"))
    monkeypatch.setattr(detector, "load", lambda: True)

    def boom(_image):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(detector, "_detect_with_model", boom)
    result = detector.detect_regions(np.zeros((40, 40, 3), dtype=np.uint8), "passport")
    assert result.backend == "full_image"
    assert result.fallback_used is True
    assert any("CUDA out of memory" in w for w in result.warnings)


def test_accepts_grayscale_and_bgra_rejects_empty(tmp_path):
    detector = DocumentRegionDetector(_config(tmp_path, fallback_mode="full_image"))
    assert detector.detect_regions(np.zeros((30, 40), dtype=np.uint8)).image_width == 40
    assert detector.detect_regions(np.zeros((30, 40, 4), dtype=np.uint8)).image_height == 30
    with pytest.raises(ValueError):
        detector.detect_regions(np.zeros((0, 0, 3), dtype=np.uint8))


def test_postprocess_predictions_thresholds_clips_and_caps(tmp_path):
    detector = DocumentRegionDetector(_config(tmp_path, max_detections_per_label=1))
    names = detector.class_names
    boxes = np.array([
        [10, 10, 60, 80],       # photo, strong
        [12, 12, 58, 78],       # photo, weaker duplicate -> capped away
        [-20, 400, 950, 470],   # mrz, overflows image -> clipped
        [5, 5, 50, 20],         # name, below threshold
        [0, 0, 10, 10],         # background -> ignored
    ], dtype=np.float32)
    labels = np.array([names.index("photo"), names.index("photo"), names.index("mrz"), names.index("name"), 0])
    scores = np.array([0.95, 0.9, 0.8, 0.1, 0.99], dtype=np.float32)

    regions = detector.postprocess_predictions(boxes, labels, scores, (450, 900, 3))
    by_label = {r.label: r for r in regions}
    assert set(by_label) == {"photo", "mrz"}
    assert by_label["photo"].bbox.confidence == pytest.approx(0.95)
    mrz = by_label["mrz"].bbox
    assert (mrz.xmin, mrz.xmax, mrz.ymax) == (0.0, 900.0, 450.0)


def test_model_backend_completeness_and_type_filter(tmp_path, monkeypatch):
    detector = DocumentRegionDetector(_config(tmp_path))
    monkeypatch.setattr(detector, "load", lambda: True)
    monkeypatch.setattr(detector, "_detect_with_model", lambda img: [
        DetectedRegion(label="mrz", bbox=make_bbox(0, 80, 100, 100, 0.9)),
        DetectedRegion(label="stamp", bbox=make_bbox(0, 0, 20, 20, 0.9)),  # not expected on a passport
    ])
    result = detector.detect_regions(np.zeros((100, 100, 3), dtype=np.uint8), "passport")
    assert result.backend == "faster_rcnn" and result.fallback_used is False
    assert [r.label for r in result.regions] == ["mrz"]
    assert "photo" in result.missing_labels
    assert result.completeness == pytest.approx(0.1)
    assert any("photo" in w for w in result.warnings)


def test_save_crops_writes_files(tmp_path):
    detector = DocumentRegionDetector(_config(tmp_path, fallback_mode="full_image", save_crops=True))
    result = detector.detect_regions(np.full((20, 30, 3), 128, dtype=np.uint8))
    path = result.regions[0].cropped_image_path
    assert path is not None
    saved = next((tmp_path / "crops").rglob("*.png"))
    assert cv2.imread(str(saved)).shape == (20, 30, 3)


def test_real_faster_rcnn_forward_pass(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    cfg = _config(tmp_path, architecture="fasterrcnn_mobilenet_v3_large_fpn", allow_untrained=True, device="cpu")
    detector = DocumentRegionDetector(cfg)
    result = detector.detect_regions(np.full((320, 480, 3), 240, dtype=np.uint8), "passport")
    assert detector.model_loaded
    assert result.backend == "faster_rcnn"
    assert result.model_version.endswith("-untrained")


# ── Geometry and schema output ─────────────────────────────────────────

def test_crop_region_basic_padding_and_clipping():
    image = np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3).astype(np.uint8)
    box = BoundingBox(xmin=10, ymin=20, xmax=60, ymax=50, confidence=0.9)

    crop = crop_region(image, box)
    assert crop.shape == (30, 50, 3)
    assert np.array_equal(crop, image[20:50, 10:60])

    assert crop_region(image, box, padding=5).shape == (40, 60, 3)
    assert crop_region(image, box, padding_ratio=0.1).shape == (36, 60, 3)

    edge = BoundingBox(xmin=-30, ymin=90, xmax=20, ymax=150, confidence=0.5)
    assert crop_region(image, edge).shape == (10, 20, 3)

    outside = BoundingBox(xmin=300, ymin=300, xmax=400, ymax=400, confidence=0.5)
    assert crop_region(image, outside).size == 0


def test_crop_region_returns_independent_copy():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    crop = crop_region(image, BoundingBox(xmin=0, ymin=0, xmax=5, ymax=5, confidence=1.0))
    crop[:] = 255
    assert image.max() == 0


def test_make_bbox_orders_and_clips():
    box = make_bbox(80, 90, 10, -5, 1.7, image_shape=(50, 60, 3))
    assert (box.xmin, box.ymin, box.xmax, box.ymax) == (10.0, 0.0, 60.0, 50.0)
    assert box.confidence == 1.0
    assert bbox_to_xywh(box) == (10, 0, 50, 50)


def test_result_schema_roundtrip_and_helpers():
    result = RegionLocalizationResult(
        regions=[
            DetectedRegion(label="photo", bbox=make_bbox(0, 0, 10, 10, 0.6)),
            DetectedRegion(label="photo", bbox=make_bbox(5, 5, 20, 20, 0.9)),
            DetectedRegion(label="document", bbox=make_bbox(0, 0, 50, 50, 0.9)),
            DetectedRegion(label="mrz", bbox=make_bbox(0, 40, 50, 50, 0.2)),
        ],
        processing_time_ms=12.5,
        model_version="test",
    )
    payload = json.loads(result.model_dump_json())
    assert set(payload["regions"][0]["bbox"]) == {"xmin", "ymin", "xmax", "ymax", "confidence"}
    assert RegionLocalizationResult(**payload) == result
    assert result.best_region("photo").bbox.confidence == 0.9
    assert result.best_region("signature") is None

    trusted = trusted_regions(result, min_confidence=0.3)
    assert [r.label for r in trusted] == ["photo", "photo"]  # fallback label and weak mrz excluded


# ── Downstream region-awareness ───────────────────────────────────────

def test_face_search_uses_photo_region_and_maps_coordinates(monkeypatch):
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    region = BoundingBox(xmin=400, ymin=100, xmax=500, ymax=220, confidence=0.9)
    calls = []

    def fake_detect(img):
        calls.append(img.shape)
        if img.shape[:2] == (400, 600):
            return []  # full image scan finds nothing
        return [face_detector.FaceBox(60, 60, 120, 120)]  # in upscaled crop coordinates

    monkeypatch.setattr(face_detector, "detect_faces", fake_detect)
    crop, box = face_detector.detect_largest_face(image, photo_region=region)
    assert box.source == "photo_region"
    # crop origin is the padded region (385, 82); crop is upscaled by 240/156
    assert 400 <= box.x <= 430 and 100 <= box.y <= 130
    assert crop.shape[:2] == (box.h, box.w)
    assert len(calls) == 1


def test_face_search_falls_back_to_full_image(monkeypatch):
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    region = BoundingBox(xmin=400, ymin=100, xmax=500, ymax=220, confidence=0.9)

    def fake_detect(img):
        return [face_detector.FaceBox(10, 10, 80, 80)] if img.shape[:2] == (400, 600) else []

    monkeypatch.setattr(face_detector, "detect_faces", fake_detect)
    _, box = face_detector.detect_largest_face(image, photo_region=region)
    assert box.source == "full_image"
    assert (box.x, box.y) == (10, 10)


def test_region_field_values_normalise_and_map():
    region_text = {
        "surname": "Surname: ERIKSSON",
        "name": "ANNA  MARIA",
        "date_of_birth": "12 AUG 1974",
        "date_of_expiry": "14.04.2032",
        "document_number": "L898 902C3",
        "nationality": "uto",
        "signature": "scribble",
    }
    values = region_field_values("passport", region_text)
    assert values == {
        "surname": "ERIKSSON",
        "given_names": "ANNA MARIA",
        "date_of_birth": "1974-08-12",
        "date_of_expiry": "2032-04-14",
        "passport_number": "L898902C3",
        "nationality": "UTO",
    }
    merged = merge_region_fields("passport", {"name": "garbled", "date_of_issue": "2012-04-15"}, region_text)
    assert merged["name"] == "ANNA MARIA ERIKSSON"
    assert merged["date_of_issue"] == "2012-04-15"  # untouched when not localized
    assert normalize_region_date("no date here") is None


def test_validate_passport_cross_checks_localized_visual_fields():
    mrz = parse_mrz(f"{VALID_LINE1}\n{VALID_LINE2}")
    baseline = validate_passport(mrz, visual_name="ANNA MARIA ERIKSSON")

    matching = validate_passport(mrz, visual_name="ANNA MARIA ERIKSSON", visual_fields={
        "passport_number": "L898902C3", "date_of_birth": "1974-08-12",
    })
    assert matching.checks_run == baseline.checks_run + 2
    assert not any("MRZ_VS_VISUAL" in i.code for i in matching.issues)

    altered = validate_passport(mrz, visual_name="ANNA MARIA ERIKSSON", visual_fields={
        "date_of_birth": "1981-02-03", "passport_number": "L898902C4",
    })
    codes = {i.code: i.severity for i in altered.issues}
    assert codes["DATE_OF_BIRTH_MISMATCH_MRZ_VS_VISUAL"] == "critical"
    assert codes["PASSPORT_NUMBER_MINOR_MISMATCH_MRZ_VS_VISUAL"] == "warning"


def test_region_tampering_flags_hot_high_risk_region():
    image = np.full((200, 300, 3), 200, dtype=np.uint8)
    cv2.putText(image, "ANNA MARIA", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, "12.08.1974", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    error_map = np.full((200, 300), 2.0, dtype=np.float32)
    error_map[20:70, 20:200] = 60.0  # re-edited name field
    ela = ELAResult(mean_error=float(error_map.mean()), max_error=60.0, suspicious_region_ratio=0.01,
                    suspicious=False, error_map=error_map, hot_threshold=25.0)
    copy_move = CopyMoveResult(match_count=0, suspicious=False)
    regions = [
        DetectedRegion(label="name", bbox=make_bbox(20, 20, 200, 70, 0.9)),
        DetectedRegion(label="date_of_birth", bbox=make_bbox(20, 120, 200, 170, 0.9)),
        DetectedRegion(label="signature", bbox=make_bbox(20, 170, 200, 199, 0.9)),  # not high-risk
        DetectedRegion(label="document_number", bbox=make_bbox(200, 120, 290, 170, 0.1)),  # untrusted
    ]
    findings = {f.label: f for f in analyze_regions(image, ela, copy_move, regions)}
    assert set(findings) == {"name", "date_of_birth"}
    assert findings["name"].suspicious is True
    assert findings["date_of_birth"].suspicious is False


def test_analyze_tampering_without_regions_is_unchanged(synthetic_passport):
    ok, buf = cv2.imencode(".jpg", synthetic_passport.image)
    data = buf.tobytes()
    plain = analyze_tampering(data, synthetic_passport.image)
    with_regions = analyze_tampering(data, synthetic_passport.image, regions=[
        DetectedRegion(label="document", bbox=make_bbox(0, 0, 900, 600, 0.9)),
    ])
    assert plain.region_findings == []
    assert with_regions.region_findings == []
    assert plain.tampering_score == with_regions.tampering_score
    assert plain.ela.error_map is None


def _clean_inputs():
    return ValidationResult(score=100, checks_run=1), TamperingResult(tampering_score=0, verdict="clean")


def test_risk_penalises_missing_critical_regions_from_trained_model():
    validation, tampering = _clean_inputs()
    baseline = compute_risk(validation, tampering)
    localization = RegionLocalizationResult(
        regions=[DetectedRegion(label="name", bbox=make_bbox(0, 0, 10, 10, 0.9))],
        model_version="test", backend="faster_rcnn",
        expected_labels=["photo", "name", "mrz"], missing_labels=["photo", "mrz"], completeness=0.333,
    )
    risk = compute_risk(validation, tampering, localization=localization, document_type="passport")
    assert risk.risk_score > baseline.risk_score
    assert risk.risk_score - baseline.risk_score <= get_settings().LOCALIZATION_RISK_MAX_POINTS
    assert risk.localization_completeness == pytest.approx(0.333)
    assert any("photo, mrz" in f for f in risk.contributing_factors)


def test_risk_ignores_fallback_localization():
    validation, tampering = _clean_inputs()
    localization = RegionLocalizationResult(model_version="heuristic-opencv-v1", backend="heuristic", fallback_used=True)
    risk = compute_risk(validation, tampering, localization=localization, document_type="passport")
    assert risk.risk_score == compute_risk(validation, tampering).risk_score


# ── Dataset loading ─────────────────────────────────────────────────────

def test_dataset_loads_synthetic_and_coco(tmp_path):
    generator = SyntheticDocumentGenerator()
    docs = generator.generate_batch("passport", count=2, output_dir=tmp_path / "synthetic")

    synthetic = DocumentRegionDataset(tmp_path / "synthetic", annotation_format="synthetic", return_tensors=False)
    assert len(synthetic) == 2
    image, target = synthetic[0]
    assert image.shape == (600, 900, 3)
    assert target["boxes"].shape == (len(docs[0].annotations), 4)
    assert (target["boxes"][:, 2] > target["boxes"][:, 0]).all()
    assert target["labels"].min() >= 1

    write_synthetic_coco(docs, tmp_path / "coco")
    coco = DocumentRegionDataset(tmp_path / "coco", annotation_format="coco", return_tensors=False)
    assert len(coco) == 2
    assert np.allclose(coco[0][1]["boxes"], target["boxes"])

    images, targets = collate_fn([synthetic[0], synthetic[1]])
    assert len(images) == 2 and len(targets) == 2


def test_dataset_loads_pascal_voc(tmp_path):
    (tmp_path / "Annotations").mkdir()
    (tmp_path / "JPEGImages").mkdir()
    cv2.imwrite(str(tmp_path / "JPEGImages" / "doc1.jpg"), np.full((100, 150, 3), 255, dtype=np.uint8))
    (tmp_path / "Annotations" / "doc1.xml").write_text(
        "<annotation><filename>doc1.jpg</filename>"
        "<object><name>photo</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>60</xmax><ymax>90</ymax></bndbox></object>"
        "<object><name>hologram</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>"
        "</annotation>",
        encoding="utf-8",
    )
    dataset = DocumentRegionDataset(tmp_path, annotation_format="voc", return_tensors=False)
    _, target = dataset[0]
    assert target["boxes"].tolist() == [[10.0, 10.0, 60.0, 90.0]]
    assert target["labels"].tolist() == [dataset.class_to_idx["photo"]]
    assert dataset.skipped_unknown_labels == 1
