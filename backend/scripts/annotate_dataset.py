#!/usr/bin/env python
"""
Auto-annotate the synthetic PAN / Passport / Aadhaar datasets for Faster
R-CNN training.

Each document type is generated from a fixed template, so region positions
are defined as fractions of the detected card boundary.  The script:

  1. Finds the card rectangle in each image (contour on white background).
  2. Maps template-relative region coordinates to absolute pixel boxes.
  3. Outputs a COCO-format ``annotations.json`` + symlinks/copies the images.
  4. Optionally saves visualisation overlays (``--visualise N``).

Usage::

    python scripts/annotate_dataset.py \
        --pan  "D:/Projects/SIH26/DATASET/pann" \
        --passport "D:/Projects/SIH26/DATASET/PASSPORT images" \
        --aadhaar "D:/Projects/SIH26/DATASET/aadhar images" \
        --output "D:/Projects/SIH26/SIH-2026/backend/data/annotations" \
        --visualise 5
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# ── Region templates ─────────────────────────────────────────────────
# (label, x_frac, y_frac, w_frac, h_frac) relative to the card boundary.
# Coordinates are intentionally generous to give the detector room.

PAN_REGIONS: list[tuple[str, float, float, float, float]] = [
    ("photo",           0.67, 0.20, 0.17, 0.72),
    ("name",            0.005, 0.19, 0.35, 0.10),
    ("surname",         0.005, 0.28, 0.35, 0.10),   # father's name
    ("date_of_birth",   0.005, 0.36, 0.25, 0.10),
    ("document_number", 0.005, 0.46, 0.25, 0.10),
    ("signature",       0.30, 0.30, 0.30, 0.10),
]

# Passport page 1 only (data page).  The image typically contains both
# pages stacked vertically; we annotate only the FIRST card.
PASSPORT_REGIONS: list[tuple[str, float, float, float, float]] = [
    ("photo",           0.01, 0.14, 0.18, 0.56),     # left main photo
    ("surname",         0.19, 0.12, 0.25, 0.07),
    ("name",            0.19, 0.19, 0.25, 0.07),
    ("nationality",     0.19, 0.26, 0.15, 0.07),
    ("date_of_birth",   0.34, 0.26, 0.15, 0.07),
    ("document_number", 0.25, 0.06, 0.25, 0.07),
    ("date_of_issue",   0.19, 0.35, 0.15, 0.07),
    ("date_of_expiry",  0.34, 0.35, 0.15, 0.07),
    ("mrz",             0.01, 0.80, 0.96, 0.17),
    ("signature",       0.02, 0.68, 0.22, 0.08),
]

AADHAAR_REGIONS: list[tuple[str, float, float, float, float]] = [
    ("photo",           0.005, 0.13, 0.18, 0.65),
    ("name",            0.20, 0.10, 0.45, 0.10),
    ("date_of_birth",   0.20, 0.22, 0.25, 0.10),
    ("document_number", 0.20, 0.32, 0.35, 0.10),
]


class CardBounds(NamedTuple):
    x: int
    y: int
    w: int
    h: int


REGION_COLORS = {
    "photo": (255, 0, 0),
    "name": (0, 200, 0),
    "surname": (0, 200, 200),
    "date_of_birth": (200, 0, 200),
    "date_of_issue": (200, 200, 0),
    "date_of_expiry": (128, 0, 128),
    "document_number": (0, 128, 255),
    "nationality": (0, 200, 128),
    "mrz": (200, 128, 0),
    "signature": (128, 128, 128),
    "stamp": (0, 0, 200),
}


# ── Card boundary detection ──────────────────────────────────────────

def find_card(image: np.ndarray, min_area_frac: float = 0.05) -> CardBounds:
    """Find the largest rectangular region against a white background."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = float(h * w)
    best = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        bx, by, bw, bh = cv2.boundingRect(c)
        if (bw * bh) / total_area < min_area_frac:
            break
        if best is None or bw * bh > best.w * best.h:
            best = CardBounds(bx, by, bw, bh)
    if best is None:
        return CardBounds(0, 0, w, h)
    return best


def find_passport_page1(image: np.ndarray) -> CardBounds:
    """Passport images contain two stacked cards.  Return the TOPMOST card
    (page 1, the data page)."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw * bh > 0.10 * h * w:
            cards.append(CardBounds(bx, by, bw, bh))
    if not cards:
        return CardBounds(0, 0, w, h)
    cards.sort(key=lambda c: c.y)
    return cards[0]


# ── Annotation generation ────────────────────────────────────────────

def regions_to_annotations(
    card: CardBounds,
    template: list[tuple[str, float, float, float, float]],
    image_shape: tuple[int, ...],
) -> list[dict]:
    """Map template fractions to absolute COCO-style [x, y, w, h] boxes."""
    ih, iw = image_shape[:2]
    annotations = []
    for label, xf, yf, wf, hf in template:
        x = max(0, int(card.x + card.w * xf))
        y = max(0, int(card.y + card.h * yf))
        w = min(int(card.w * wf), iw - x)
        h = min(int(card.h * hf), ih - y)
        if w < 5 or h < 5:
            continue
        annotations.append({"label": label, "bbox": [x, y, w, h]})
    return annotations


def visualise(image: np.ndarray, annotations: list[dict]) -> np.ndarray:
    vis = image.copy()
    for ann in annotations:
        label = ann["label"]
        x, y, w, h = ann["bbox"]
        color = REGION_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
        cv2.putText(vis, label, (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


# ── COCO dataset builder ─────────────────────────────────────────────

CLASS_NAMES = [
    "background", "photo", "name", "surname", "date_of_birth",
    "date_of_issue", "date_of_expiry", "document_number",
    "nationality", "mrz", "signature", "stamp",
]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES) if n != "background"}


def build_coco_dataset(
    datasets: list[tuple[str, Path, list[tuple[str, float, float, float, float]]]],
    output_dir: Path,
    vis_count: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)
    vis_dir = output_dir / "visualisations"
    if vis_count > 0:
        vis_dir.mkdir(exist_ok=True)

    categories = [{"id": v, "name": k} for k, v in CLASS_TO_ID.items()]
    coco_images = []
    coco_annotations = []
    ann_id = 1
    image_id = 0
    vis_saved = 0

    for doc_type, source_dir, template in datasets:
        if not source_dir.is_dir():
            print(f"  SKIP {doc_type}: {source_dir} not found")
            continue
        files = sorted(
            p for p in source_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        print(f"  {doc_type}: {len(files)} images in {source_dir}")

        for fpath in files:
            image_id += 1
            image = cv2.imread(str(fpath))
            if image is None:
                print(f"    WARN: could not read {fpath.name}")
                continue
            ih, iw = image.shape[:2]

            if doc_type == "passport":
                card = find_passport_page1(image)
            else:
                card = find_card(image)

            annotations = regions_to_annotations(card, template, image.shape)

            dest_name = f"{doc_type}_{image_id:05d}{fpath.suffix.lower()}"
            dest_path = img_dir / dest_name
            shutil.copy2(fpath, dest_path)

            coco_images.append({
                "id": image_id,
                "file_name": dest_name,
                "width": iw,
                "height": ih,
            })
            for ann in annotations:
                cat_id = CLASS_TO_ID.get(ann["label"])
                if cat_id is None:
                    continue
                x, y, w, h = ann["bbox"]
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

            if vis_saved < vis_count:
                vis_img = visualise(image, annotations)
                cv2.imwrite(str(vis_dir / f"vis_{dest_name}"), vis_img)
                vis_saved += 1

    ann_path = output_dir / "annotations.json"
    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"\nCOCO dataset written:")
    print(f"  images:      {len(coco_images)}")
    print(f"  annotations: {len(coco_annotations)}")
    print(f"  categories:  {len(categories)}")
    print(f"  output:      {ann_path}")
    if vis_count > 0:
        print(f"  visualisations: {vis_dir} ({vis_saved} images)")


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate synthetic document datasets.")
    parser.add_argument("--pan", type=Path, help="Path to PAN card image directory")
    parser.add_argument("--passport", type=Path, help="Path to passport image directory")
    parser.add_argument("--aadhaar", type=Path, help="Path to Aadhaar image directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for COCO dataset")
    parser.add_argument("--visualise", type=int, default=0, help="Save N visualisation overlays")
    args = parser.parse_args()

    datasets: list[tuple[str, Path, list]] = []
    if args.pan:
        datasets.append(("pan_card", args.pan, PAN_REGIONS))
    if args.passport:
        datasets.append(("passport", args.passport, PASSPORT_REGIONS))
    if args.aadhaar:
        datasets.append(("aadhaar", args.aadhaar, AADHAAR_REGIONS))

    if not datasets:
        print("No datasets specified. Use --pan, --passport, and/or --aadhaar.")
        sys.exit(1)

    print(f"Annotating {len(datasets)} dataset(s)...")
    build_coco_dataset(datasets, args.output, vis_count=args.visualise)


if __name__ == "__main__":
    main()
