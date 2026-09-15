#!/usr/bin/env python
"""
Train Faster R-CNN on the annotated synthetic document dataset.

The synthetic images are template-generated (zero layout variation), so heavy
augmentation is essential: perspective warp, rotation, colour jitter, noise,
and blur simulate the kind of variation seen in real scans and photos.

Saves checkpoints in the format ``detector.py`` expects::

    {
        "model_state_dict": ...,
        "class_names": [...],
        "model_version": "fasterrcnn_resnet50_fpn-docregions-v1",
        "epoch": N,
        "best_loss": float,
    }

Usage::

    python scripts/train_rcnn.py \
        --data data/annotations \
        --epochs 30 \
        --batch-size 4 \
        --lr 0.005

The ``--output`` directory defaults to ``models/localization/``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running from the backend/ directory or project root.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND = _SCRIPT_DIR.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.modules.localization.config import REGION_CLASSES


# ── Augmentation ────────────────────────────────────────────────────────
# All transforms operate on (image_bgr, boxes_xyxy) and return the same.

def _clamp_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def _remove_degenerate(boxes: np.ndarray, labels: np.ndarray, min_side: float = 4.0):
    if len(boxes) == 0:
        return boxes, labels
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths >= min_side) & (heights >= min_side)
    return boxes[keep], labels[keep]


def augment_hsv(image: np.ndarray, h_gain: float = 0.015, s_gain: float = 0.7, v_gain: float = 0.4):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_noise(image: np.ndarray, sigma_range: tuple[float, float] = (5.0, 25.0)):
    sigma = np.random.uniform(*sigma_range)
    noise = np.random.randn(*image.shape) * sigma
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_blur(image: np.ndarray, max_k: int = 5):
    k = random.choice(range(1, max_k + 1, 2))
    return cv2.GaussianBlur(image, (k, k), 0)


def augment_perspective(
    image: np.ndarray,
    boxes: np.ndarray,
    max_warp: float = 0.03,
):
    h, w = image.shape[:2]
    margin = max_warp
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([
        [random.uniform(0, margin) * w, random.uniform(0, margin) * h],
        [random.uniform(-margin, 0) * w, random.uniform(0, margin) * h],
        [random.uniform(-margin, 0) * w, random.uniform(-margin, 0) * h],
        [random.uniform(0, margin) * w, random.uniform(-margin, 0) * h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h), borderValue=(128, 128, 128))

    if len(boxes) == 0:
        return warped, boxes

    corners = []
    for bx in boxes:
        pts = np.float32([
            [bx[0], bx[1]], [bx[2], bx[1]],
            [bx[2], bx[3]], [bx[0], bx[3]],
        ]).reshape(-1, 1, 2)
        corners.append(pts)
    all_corners = np.concatenate(corners, axis=0)
    transformed = cv2.perspectiveTransform(all_corners, M)

    new_boxes = []
    for i in range(len(boxes)):
        pts = transformed[i * 4:(i + 1) * 4].reshape(4, 2)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        new_boxes.append([x0, y0, x1, y1])
    new_boxes = np.array(new_boxes, dtype=np.float32)
    return warped, _clamp_boxes(new_boxes, w, h)


def augment_rotate(
    image: np.ndarray,
    boxes: np.ndarray,
    max_angle: float = 5.0,
):
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(128, 128, 128))

    if len(boxes) == 0:
        return rotated, boxes

    corners = []
    for bx in boxes:
        pts = np.array([
            [bx[0], bx[1], 1], [bx[2], bx[1], 1],
            [bx[2], bx[3], 1], [bx[0], bx[3], 1],
        ], dtype=np.float32)
        corners.append(pts)
    all_corners = np.concatenate(corners, axis=0)
    transformed = (M @ all_corners.T).T  # (N*4, 2)

    new_boxes = []
    for i in range(len(boxes)):
        pts = transformed[i * 4:(i + 1) * 4]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        new_boxes.append([x0, y0, x1, y1])
    new_boxes = np.array(new_boxes, dtype=np.float32)
    return rotated, _clamp_boxes(new_boxes, w, h)


def augment_flip_h(image: np.ndarray, boxes: np.ndarray):
    w = image.shape[1]
    flipped = cv2.flip(image, 1)
    if len(boxes) == 0:
        return flipped, boxes
    new_boxes = boxes.copy()
    new_boxes[:, 0] = w - boxes[:, 2]
    new_boxes[:, 2] = w - boxes[:, 0]
    return flipped, new_boxes


def apply_augmentations(image: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
    if random.random() < 0.5:
        image, boxes = augment_perspective(image, boxes)
    if random.random() < 0.5:
        image, boxes = augment_rotate(image, boxes)
    if random.random() < 0.15:
        image, boxes = augment_flip_h(image, boxes)
    if random.random() < 0.8:
        image = augment_hsv(image)
    if random.random() < 0.3:
        image = augment_noise(image)
    if random.random() < 0.3:
        image = augment_blur(image)
    boxes, labels = _remove_degenerate(boxes, labels)
    return image, boxes, labels


# ── Dataset wrapper with augmentation ───────────────────────────────────

class AugmentedDataset:
    """Wraps DocumentRegionDataset, applying augmentation at __getitem__ time."""

    def __init__(self, base_dataset, augment: bool = True):
        self.base = base_dataset
        self.augment = augment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        import torch

        record = self.base.records[index]
        image_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read {record.image_path}")
        h, w = image_bgr.shape[:2]

        boxes = np.asarray(record.boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(record.labels, dtype=np.int64)

        if self.augment and len(boxes) > 0:
            image_bgr, boxes, labels = apply_augmentations(image_bgr, boxes, labels)
            h, w = image_bgr.shape[:2]

        boxes = _clamp_boxes(boxes, w, h)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else np.zeros(0, dtype=np.float32)
        iscrowd = np.zeros(len(labels), dtype=np.int64)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor(record.image_id, dtype=torch.int64),
            "area": torch.from_numpy(areas.astype(np.float32)),
            "iscrowd": torch.from_numpy(iscrowd),
        }
        return image_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# ── Training loop ───────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, data_loader, device, epoch: int, print_freq: int = 20):
    import torch
    model.train()
    running_loss = 0.0
    count = 0
    t0 = time.perf_counter()

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if all(t["boxes"].shape[0] == 0 for t in targets):
            continue

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"  WARNING: non-finite loss {loss_value}, skipping batch {i}")
            continue

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += loss_value
        count += 1

        if (i + 1) % print_freq == 0:
            elapsed = time.perf_counter() - t0
            avg = running_loss / count if count else 0
            print(f"  [epoch {epoch}] batch {i+1}/{len(data_loader)}  "
                  f"avg_loss={avg:.4f}  elapsed={elapsed:.1f}s")

    return running_loss / count if count else float("inf")


def evaluate(model, data_loader, device):
    import torch
    model.train()  # Faster R-CNN returns loss_dict only in train mode
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if all(t["boxes"].shape[0] == 0 for t in targets):
                continue
            loss_dict = model(images, targets)
            total_loss += float(sum(loss_dict.values()))
            count += 1
    return total_loss / count if count else float("inf")


def build_model(num_classes: int, architecture: str = "fasterrcnn_resnet50_fpn"):
    from torchvision.models import detection as tv_detection
    builder = getattr(tv_detection, architecture)
    model = builder(weights="DEFAULT", weights_backbone="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def save_checkpoint(model, class_names, model_version, epoch, loss, path: Path):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "model_version": model_version,
        "epoch": epoch,
        "best_loss": loss,
    }, path)
    print(f"  Checkpoint saved: {path}  (epoch {epoch}, loss {loss:.4f})")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN document region detector")
    parser.add_argument("--data", type=Path, required=True,
                        help="COCO dataset directory (annotations.json + images/)")
    parser.add_argument("--output", type=Path, default=_BACKEND / "models" / "localization",
                        help="Directory for checkpoints")
    parser.add_argument("--architecture", default="fasterrcnn_resnet50_fpn",
                        choices=["fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2",
                                 "fasterrcnn_mobilenet_v3_large_fpn"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of data for validation")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (0 = main process)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import torch
        from torch.utils.data import DataLoader, Subset
    except ImportError:
        print("ERROR: PyTorch is required. Install with:")
        print("  pip install torch torchvision")
        sys.exit(1)

    torch.manual_seed(args.seed)

    # Load base dataset (without tensors — we handle that in AugmentedDataset)
    from app.modules.localization.dataset import DocumentRegionDataset
    print(f"Loading dataset from {args.data} ...")
    base_ds = DocumentRegionDataset(
        root=args.data,
        annotation_format="coco",
        return_tensors=False,
        skip_empty=True,
    )
    print(f"  {len(base_ds)} images, {len(base_ds.class_names)} classes: {base_ds.class_names}")

    # Train/val split
    n = len(base_ds)
    indices = list(range(n))
    random.shuffle(indices)
    n_val = max(1, int(n * args.val_split))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    print(f"  train: {len(train_indices)}, val: {len(val_indices)}")

    train_ds = AugmentedDataset(Subset(base_ds, train_indices), augment=True)
    val_ds = AugmentedDataset(Subset(base_ds, val_indices), augment=False)

    # Subset wrapping: AugmentedDataset needs .records on its base
    # Patch: make Subset-backed AugmentedDataset work by forwarding record access
    class SubsetAdapter:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
            self.records = [dataset.records[i] for i in indices]
        def __len__(self):
            return len(self.indices)

    train_ds.base = SubsetAdapter(base_ds, train_indices)
    val_ds.base = SubsetAdapter(base_ds, val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
    )

    # Model
    class_names = list(base_ds.class_names)
    num_classes = len(class_names)
    model_version = f"{args.architecture}-docregions-v1"
    print(f"Building {args.architecture} with {num_classes} classes ...")
    model = build_model(num_classes, args.architecture)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")

    # Optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    start_epoch = 0
    best_loss = float("inf")
    if args.resume and args.resume.is_file():
        print(f"Resuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best_loss={best_loss:.4f}")

    # Checkpoint paths
    best_path = args.output / "fasterrcnn_document_regions.pth"
    last_path = args.output / "fasterrcnn_document_regions_last.pth"

    # Training
    patience_counter = 0
    print(f"\nTraining for {args.epochs} epochs (patience={args.patience}) ...\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, optimizer, train_loader, device,
                                     epoch, print_freq=max(1, len(train_loader) // 3))
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs-1}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  lr={lr:.6f}  time={elapsed:.1f}s")

        save_checkpoint(model, class_names, model_version, epoch, val_loss, last_path)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, class_names, model_version, epoch, val_loss, best_path)
            print(f"  ** New best model (val_loss={best_loss:.4f}) **")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"\nTraining complete. Best val_loss={best_loss:.4f}")
    print(f"  Best checkpoint:  {best_path}")
    print(f"  Last checkpoint:  {last_path}")


if __name__ == "__main__":
    main()
