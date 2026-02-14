#!/usr/bin/env python3
"""
Export a YOLO-ready dataset from cleaned UAVDT (Dataset-Ninja) vehicle annotations.

Inputs:
  - Images:       data/processed/uavdt_vehicle_clean/images/
  - Annotations:  data/processed/uavdt_vehicle_clean/annotations/
    Each annotation JSON contains:
      - objects: list
      - object.points.exterior = [[x1,y1],[x2,y2]] (top-left, bottom-right)
      - object.cls = 0  (already unified to vehicle)

Output (created):
  data/processed/uavdt_vehicle_yolo/
    ├── images/train
    ├── images/val
    ├── labels/train
    └── labels/val

Rules:
  - Deterministic split: 90% train, 10% val (seed=42)
  - Map annotation -> image by removing ".json" then searching stem.* in images dir
  - Read image size from actual file (PIL)
  - Validate boxes (ordering, clip, positive area)
  - Convert to YOLO normalized format: cls x_center y_center w h
  - Hard cap AFTER validation: if valid boxes > bbox_cap -> skip the image entirely
  - If valid boxes == 0 -> skip the image (no empty-label exports)

Print final summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.90
DEFAULT_BBOX_CAP = 25
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("uavdt_yolo_export")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    h = logging.StreamHandler()
    h.setLevel(logging.DEBUG if verbose else logging.INFO)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO-ready dataset from cleaned UAVDT vehicle annotations.")
    p.add_argument("--images", default="data/processed/uavdt_vehicle_clean/images", help="Input images directory")
    p.add_argument("--annotations", default="data/processed/uavdt_vehicle_clean/annotations", help="Input annotations directory")
    p.add_argument("--out", default="data/processed/uavdt_vehicle_yolo", help="Output YOLO dataset root directory")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for deterministic train/val split")
    p.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Train split ratio (val=1-train)")
    p.add_argument("--bbox-cap", type=int, default=DEFAULT_BBOX_CAP, help="Skip images with more than this many VALID boxes")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Annotation root must be a dict: {path}")
    return data


def ensure_dirs(out_root: Path) -> Dict[str, Path]:
    paths = {
        "images_train": out_root / "images" / "train",
        "images_val": out_root / "images" / "val",
        "labels_train": out_root / "labels" / "train",
        "labels_val": out_root / "labels" / "val",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def get_image_size(img_path: Path) -> Tuple[int, int]:
    with Image.open(img_path) as im:
        w, h = im.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {img_path} -> {w}x{h}")
    return w, h


def clip_bbox_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    x1c = max(0.0, min(float(w), x1))
    y1c = max(0.0, min(float(h), y1))
    x2c = max(0.0, min(float(w), x2))
    y2c = max(0.0, min(float(h), y2))
    return x1c, y1c, x2c, y2c


def xyxy_to_yolo_norm(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    xcn = xc / float(w)
    ycn = yc / float(h)
    bwn = bw / float(w)
    bhn = bh / float(h)

    def clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    return clamp01(xcn), clamp01(ycn), clamp01(bwn), clamp01(bhn)


def extract_boxes_yolo_lines(ann: Dict[str, Any], img_w: int, img_h: int, logger: logging.Logger) -> List[str]:
    objs = ann.get("objects", [])
    if not isinstance(objs, list):
        return []

    lines: List[str] = []
    for i, obj in enumerate(objs):
        if not isinstance(obj, dict):
            continue

        pts = (obj.get("points") or {}).get("exterior")
        if (
            not isinstance(pts, list)
            or len(pts) != 2
            or not isinstance(pts[0], list)
            or not isinstance(pts[1], list)
            or len(pts[0]) != 2
            or len(pts[1]) != 2
        ):
            logger.debug(f"Skipping malformed bbox points in object #{i}")
            continue

        x1, y1 = float(pts[0][0]), float(pts[0][1])
        x2, y2 = float(pts[1][0]), float(pts[1][1])

        # Ensure ordering
        if not (x1 < x2 and y1 < y2):
            logger.debug(f"Skipping invalid ordered bbox in object #{i}: ({x1},{y1})-({x2},{y2})")
            continue

        # Clip to boundaries
        x1c, y1c, x2c, y2c = clip_bbox_xyxy(x1, y1, x2, y2, img_w, img_h)

        # Positive area after clipping
        if not (x1c < x2c and y1c < y2c):
            logger.debug(f"Skipping non-positive clipped bbox in object #{i}: ({x1c},{y1c})-({x2c},{y2c})")
            continue

        xcn, ycn, bwn, bhn = xyxy_to_yolo_norm(x1c, y1c, x2c, y2c, img_w, img_h)

        # cls forced to 0 (vehicle)
        line = f"0 {xcn:.6f} {ycn:.6f} {bwn:.6f} {bhn:.6f}"
        lines.append(line)

    return lines


def build_image_index(images_dir: Path) -> Dict[str, Path]:
    """
    Map filename (with extension) and stem to image path for robust lookup.
    We prioritize exact name match (after removing .json) then fallback to stem match.
    """
    idx: Dict[str, Path] = {}
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            idx[p.name] = p
            idx[p.stem] = p
    return idx


def annotation_to_image_path(image_index: Dict[str, Path], ann_name: str) -> Optional[Path]:
    """
    ann_name is like: M0203_img000001.jpg.json
    We remove only the trailing ".json" -> M0203_img000001.jpg
    Then try exact filename match; fallback to stem match.
    """
    if not ann_name.endswith(".json"):
        return None
    base = ann_name[:-len(".json")]  # keeps .jpg if present
    if base in image_index:
        return image_index[base]
    # fallback: maybe annotation base has an extension but image differs; use stem
    stem = Path(base).stem
    return image_index.get(stem)


def deterministic_split(names: List[str], seed: int, train_ratio: float) -> Tuple[set[str], set[str]]:
    rnd = random.Random(seed)
    items = list(names)
    rnd.shuffle(items)
    n = len(items)
    n_train = int(math.floor(n * train_ratio))
    return set(items[:n_train]), set(items[n_train:])


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.verbose)

    in_images = Path(args.images)
    in_anns = Path(args.annotations)
    out_root = Path(args.out)

    if not in_images.is_dir():
        raise SystemExit(f"Input images directory not found: {in_images}")
    if not in_anns.is_dir():
        raise SystemExit(f"Input annotations directory not found: {in_anns}")

    out_paths = ensure_dirs(out_root)

    ann_files = sorted(p for p in in_anns.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    total_items_scanned = len(ann_files)

    ann_names = [p.name for p in ann_files]
    train_set, val_set = deterministic_split(ann_names, args.seed, args.train_ratio)

    logger.info("Indexing images for fast lookup...")
    image_index = build_image_index(in_images)
    logger.info(f"Indexed keys: {len(image_index)}")

    exported_train_images = 0
    exported_val_images = 0
    skipped_bbox_cap = 0
    skipped_missing_image = 0
    skipped_empty_after_validation = 0
    total_boxes_exported = 0

    for ann_path in ann_files:
        ann_name = ann_path.name
        img_path = annotation_to_image_path(image_index, ann_name)

        if img_path is None or not img_path.exists():
            skipped_missing_image += 1
            continue

        try:
            ann = read_json(ann_path)
        except Exception as e:
            logger.warning(f"Skipping bad JSON: {ann_name} ({e})")
            continue

        try:
            img_w, img_h = get_image_size(img_path)
        except Exception as e:
            skipped_missing_image += 1
            logger.warning(f"Skipping unreadable image: {img_path.name} ({e})")
            continue

        yolo_lines = extract_boxes_yolo_lines(ann, img_w, img_h, logger)

        # Skip if no valid boxes after validation (keep pipeline strict for now)
        if len(yolo_lines) == 0:
            skipped_empty_after_validation += 1
            continue

        # Hard cap AFTER validation (critical)
        if len(yolo_lines) > int(args.bbox_cap):
            skipped_bbox_cap += 1
            continue

        is_train = ann_name in train_set
        img_out_dir = out_paths["images_train"] if is_train else out_paths["images_val"]
        lbl_out_dir = out_paths["labels_train"] if is_train else out_paths["labels_val"]

        shutil.copy2(img_path, img_out_dir / img_path.name)

        label_name = img_path.with_suffix(".txt").name
        with (lbl_out_dir / label_name).open("w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        if is_train:
            exported_train_images += 1
        else:
            exported_val_images += 1
        total_boxes_exported += len(yolo_lines)

    print("Summary")
    print(f"  total items scanned:           {total_items_scanned}")
    print(f"  exported train images:         {exported_train_images}")
    print(f"  exported val images:           {exported_val_images}")
    print(f"  skipped due to bbox_cap:       {skipped_bbox_cap}")
    print(f"  skipped due to missing image:  {skipped_missing_image}")
    print(f"  skipped empty after validation:{skipped_empty_after_validation}")
    print(f"  total boxes exported:          {total_boxes_exported}")


if __name__ == "__main__":
    main()
