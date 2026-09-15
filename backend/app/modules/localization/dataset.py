"""
Dataset helpers for fine-tuning the Faster R-CNN document region detector.

Supported annotation formats:
  - "coco":      one instances JSON (bbox = [x, y, w, h]) plus an image folder
  - "voc":       Pascal VOC XML files (bndbox = xmin/ymin/xmax/ymax)
  - "synthetic": the `<stem>.png` + `<stem>_annotations.json` pairs written by
                 `SyntheticDocumentGenerator.generate_batch`

`DocumentRegionDataset` is a map-style dataset (`__len__` / `__getitem__`),
so it plugs straight into `torch.utils.data.DataLoader` with `collate_fn`.
Samples use torchvision's detection target format: boxes as
[xmin, ymin, xmax, ymax] floats and labels as class indices where 0 is
background. torch is only needed when `return_tensors=True`.
"""
from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import cv2
import numpy as np

from app.modules.localization.config import REGION_CLASSES

logger = logging.getLogger(__name__)

AnnotationFormat = Literal["coco", "voc", "synthetic"]
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class AnnotationRecord:
    image_path: Path
    image_id: int
    boxes: list[list[float]] = field(default_factory=list)  # [xmin, ymin, xmax, ymax]
    labels: list[int] = field(default_factory=list)
    iscrowd: list[int] = field(default_factory=list)


class DocumentRegionDataset:
    def __init__(
        self,
        root: str | Path,
        annotation_format: AnnotationFormat = "coco",
        annotation_file: str | Path | None = None,
        images_dir: str | Path | None = None,
        class_names: list[str] | None = None,
        return_tensors: bool = True,
        skip_empty: bool = False,
    ):
        self.root = Path(root)
        self.annotation_format = annotation_format
        self.class_names = list(class_names or REGION_CLASSES)
        if self.class_names[0] != "background":
            self.class_names.insert(0, "background")
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.return_tensors = return_tensors
        self.skipped_unknown_labels = 0

        if annotation_format == "coco":
            ann_file = Path(annotation_file) if annotation_file else self.root / "annotations.json"
            img_dir = Path(images_dir) if images_dir else self.root / "images"
            records = self._load_coco(ann_file, img_dir)
        elif annotation_format == "voc":
            ann_dir = Path(annotation_file) if annotation_file else self.root / "Annotations"
            img_dir = Path(images_dir) if images_dir else self.root / "JPEGImages"
            records = self._load_voc(ann_dir, img_dir)
        elif annotation_format == "synthetic":
            records = self._load_synthetic(Path(images_dir) if images_dir else self.root)
        else:
            raise ValueError(f"Unsupported annotation format: {annotation_format}")

        if skip_empty:
            records = [r for r in records if r.boxes]
        self.records = records
        if self.skipped_unknown_labels:
            logger.warning(
                "Skipped %d annotations with labels outside the class list.", self.skipped_unknown_labels
            )

    # -- loaders -------------------------------------------------------------

    def _label_index(self, name: str) -> int | None:
        idx = self.class_to_idx.get(name)
        if idx is None or idx == 0:
            self.skipped_unknown_labels += 1
            return None
        return idx

    @staticmethod
    def _valid_box(x0: float, y0: float, x1: float, y1: float) -> bool:
        return x1 - x0 >= 1.0 and y1 - y0 >= 1.0

    def _load_coco(self, ann_file: Path, img_dir: Path) -> list[AnnotationRecord]:
        with open(ann_file, encoding="utf-8") as f:
            coco = json.load(f)

        category_names = {c["id"]: c["name"] for c in coco.get("categories", [])}
        records: dict[int, AnnotationRecord] = {}
        for img in coco.get("images", []):
            image_id = int(img["id"])
            records[image_id] = AnnotationRecord(image_path=img_dir / img["file_name"], image_id=image_id)

        for ann in coco.get("annotations", []):
            record = records.get(int(ann["image_id"]))
            if record is None:
                continue
            idx = self._label_index(category_names.get(ann["category_id"], ""))
            if idx is None:
                continue
            x, y, w, h = (float(v) for v in ann["bbox"])
            if not self._valid_box(x, y, x + w, y + h):
                continue
            record.boxes.append([x, y, x + w, y + h])
            record.labels.append(idx)
            record.iscrowd.append(int(ann.get("iscrowd", 0)))

        return [records[k] for k in sorted(records)]

    def _load_voc(self, ann_dir: Path, img_dir: Path) -> list[AnnotationRecord]:
        records: list[AnnotationRecord] = []
        for image_id, xml_path in enumerate(sorted(ann_dir.glob("*.xml"))):
            tree = ET.parse(xml_path).getroot()
            filename = tree.findtext("filename") or f"{xml_path.stem}.jpg"
            record = AnnotationRecord(image_path=img_dir / filename, image_id=image_id)
            for obj in tree.findall("object"):
                idx = self._label_index((obj.findtext("name") or "").strip())
                box = obj.find("bndbox")
                if idx is None or box is None:
                    continue
                x0, y0, x1, y1 = (float(box.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
                if not self._valid_box(x0, y0, x1, y1):
                    continue
                record.boxes.append([x0, y0, x1, y1])
                record.labels.append(idx)
                record.iscrowd.append(int(obj.findtext("difficult", "0") or 0))
            records.append(record)
        return records

    def _load_synthetic(self, directory: Path) -> list[AnnotationRecord]:
        records: list[AnnotationRecord] = []
        for image_id, ann_path in enumerate(sorted(directory.glob("*_annotations.json"))):
            stem = ann_path.name[: -len("_annotations.json")]
            image_path = next(
                (directory / f"{stem}{s}" for s in _IMAGE_SUFFIXES if (directory / f"{stem}{s}").exists()),
                None,
            )
            if image_path is None:
                logger.warning("No image found for %s", ann_path.name)
                continue
            with open(ann_path, encoding="utf-8") as f:
                annotations = json.load(f)
            record = AnnotationRecord(image_path=image_path, image_id=image_id)
            for ann in annotations:
                idx = self._label_index(ann.get("class_name", ""))
                if idx is None:
                    continue
                x, y, w, h = (float(v) for v in ann["bbox"])
                if not self._valid_box(x, y, x + w, y + h):
                    continue
                record.boxes.append([x, y, x + w, y + h])
                record.labels.append(idx)
                record.iscrowd.append(0)
            records.append(record)
        return records

    # -- dataset protocol ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        record = self.records[index]
        image_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image {record.image_path}")
        h, w = image_bgr.shape[:2]

        boxes = np.asarray(record.boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        labels = np.asarray(record.labels, dtype=np.int64)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = np.asarray(record.iscrowd or [0] * len(labels), dtype=np.int64)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if not self.return_tensors:
            return rgb, {
                "boxes": boxes, "labels": labels, "image_id": record.image_id,
                "area": areas, "iscrowd": iscrowd,
            }

        import torch

        image_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor(record.image_id, dtype=torch.int64),
            "area": torch.from_numpy(areas.astype(np.float32)),
            "iscrowd": torch.from_numpy(iscrowd),
        }
        return image_tensor, target


def collate_fn(batch: Iterable[tuple[Any, dict[str, Any]]]) -> tuple[tuple, tuple]:
    """Detection batches have variable box counts, so keep them as tuples."""
    images, targets = zip(*batch)
    return images, targets


def write_synthetic_coco(
    documents: Iterable[Any],
    output_dir: str | Path,
    class_names: list[str] | None = None,
) -> Path:
    """
    Write `SyntheticDocument` objects as a COCO dataset (`images/` +
    `annotations.json`) that `DocumentRegionDataset(root, "coco")` can load.
    """
    names = list(class_names or REGION_CLASSES)
    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    categories = [{"id": i, "name": n} for i, n in enumerate(names) if n != "background"]
    cat_ids = {c["name"]: c["id"] for c in categories}
    images, annotations = [], []
    ann_id = 1
    for image_id, doc in enumerate(documents, start=1):
        file_name = f"{doc.document_type}_{image_id:05d}.png"
        cv2.imwrite(str(img_dir / file_name), doc.image)
        h, w = doc.image.shape[:2]
        images.append({"id": image_id, "file_name": file_name, "width": w, "height": h})
        for ann in doc.annotations:
            if ann.class_name not in cat_ids:
                continue
            x, y, bw, bh = (float(v) for v in ann.bbox)
            annotations.append({
                "id": ann_id, "image_id": image_id, "category_id": cat_ids[ann.class_name],
                "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0,
            })
            ann_id += 1

    ann_path = out / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    return ann_path
