#!/usr/bin/env python3
"""
UAVDT Dataset-Ninja annotation cleaner (competition-grade dataset pipeline step).

This script:
- Reads Dataset-Ninja JSON annotations from:
    data/raw/uavdt_dn/uavdt/train/ann/
- Finds corresponding images in:
    data/raw/uavdt_dn/uavdt/train/img/
- Keeps ONLY objects whose classTitle is in {"car", "truck", "bus"}
- Maps all kept objects to a single class: cls = 0 (Taşıt)
- Skips images with zero remaining objects after filtering
- Copies kept images to:
    data/processed/uavdt_vehicle_clean/images/
- Writes cleaned annotations (JSON, not YOLO) to:
    data/processed/uavdt_vehicle_clean/annotations/

Design notes:
- Deterministic processing (sorted file list)
- RAW dataset is never modified
- No resizing, no augmentation, no format conversion
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


KEEP_CLASS_TITLES = {"car", "truck", "bus"}
MAPPED_CLS_ID = 0  # Taşıt


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("uavdt_vehicle_cleaner")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.addHandler(handler)
    return logger


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Annotation root must be a dict: {path}")
    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_corresponding_image(img_dir: Path, stem: str) -> Optional[Path]:
    candidates = sorted(img_dir.glob(f"{stem}.*"))
    for p in candidates:
        if p.is_file() and p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
        }:
            return p
    return None


def filter_objects(objects: Any) -> Tuple[List[Dict[str, Any]], int]:
    if not isinstance(objects, list):
        return [], 0

    filtered: List[Dict[str, Any]] = []
    kept = 0

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        if obj.get("classTitle") in KEEP_CLASS_TITLES:
            new_obj = dict(obj)
            new_obj["cls"] = MAPPED_CLS_ID
            filtered.append(new_obj)
            kept += 1

    return filtered, kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare UAVDT dataset for TEKNOFEST (vehicles only, cls=0)."
    )
    parser.add_argument(
        "--img-dir",
        default="data/raw/uavdt_dn/uavdt/train/img",
        help="Input image directory",
    )
    parser.add_argument(
        "--ann-dir",
        default="data/raw/uavdt_dn/uavdt/train/ann",
        help="Input annotation directory",
    )
    parser.add_argument(
        "--out-images",
        default="data/processed/uavdt_vehicle_clean/images",
        help="Output image directory",
    )
    parser.add_argument(
        "--out-annotations",
        default="data/processed/uavdt_vehicle_clean/annotations",
        help="Output annotation directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.verbose)

    img_dir = Path(args.img_dir)
    ann_dir = Path(args.ann_dir)
    out_images = Path(args.out_images)
    out_annotations = Path(args.out_annotations)

    logger.info(f"Input images:      {img_dir}")
    logger.info(f"Input annotations:{ann_dir}")
    logger.info(f"Output images:    {out_images}")
    logger.info(f"Output annotations:{out_annotations}")

    if not img_dir.is_dir():
        raise SystemExit(f"Image directory not found: {img_dir}")
    if not ann_dir.is_dir():
        raise SystemExit(f"Annotation directory not found: {ann_dir}")

    out_images.mkdir(parents=True, exist_ok=True)
    out_annotations.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(p for p in ann_dir.iterdir() if p.suffix.lower() == ".json")

    total_images = 0
    images_kept = 0
    total_bboxes = 0

    for ann_path in ann_files:
        total_images += 1
        stem = ann_path.name.replace(".json", "").replace(".jpg", "")

        img_path = find_corresponding_image(img_dir, stem)
        if img_path is None:
            logger.warning(f"Missing image for annotation: {ann_path.name}")
            continue

        try:
            ann = read_json(ann_path)
        except Exception as e:
            logger.warning(f"Invalid annotation JSON: {ann_path.name} ({e})")
            continue

        filtered_objs, kept = filter_objects(ann.get("objects"))
        if kept == 0:
            continue

        cleaned = dict(ann)
        cleaned["objects"] = filtered_objs

        shutil.copy2(img_path, out_images / img_path.name)
        write_json(out_annotations / ann_path.name, cleaned)

        images_kept += 1
        total_bboxes += kept

    print("\nSummary")
    print(f"  total images processed: {total_images}")
    print(f"  images kept:           {images_kept}")
    print(f"  total bboxes kept:     {total_bboxes}")


if __name__ == "__main__":
    main()
