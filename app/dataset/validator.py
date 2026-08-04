from __future__ import annotations

import hashlib
import math
from pathlib import Path

from PIL import Image

from app.dataset.detector import IMAGE_SUFFIXES, DatasetLayout
from app.dataset.report import DatasetReport


def _images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _validate_label(path: Path, classes: list[str], report: DatasetReport) -> None:
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw.split()
        if len(fields) != 5:
            report.add("fatal", "field_count", path, "Expected exactly 5 fields", number)
            continue
        try:
            class_id = int(fields[0])
            if str(class_id) != fields[0] or class_id < 0 or class_id >= len(classes):
                raise ValueError
        except ValueError:
            report.add("fatal", "class_id", path, "Invalid or unknown class_id", number)
            continue
        try:
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError:
            report.add("fatal", "bbox_number", path, "Bounding box values must be floats", number)
            continue
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            report.add("fatal", "bbox_finite", path, "Bounding box values must be finite", number)
        elif not 0 <= x <= 1 or not 0 <= y <= 1 or not 0 < width <= 1 or not 0 < height <= 1:
            report.add("fatal", "bbox_range", path, "Bounding box values outside normalized range", number)
        elif x - width / 2 < 0 or y - height / 2 < 0 or x + width / 2 > 1 or y + height / 2 > 1:
            report.add("fatal", "bbox_bounds", path, "Bounding box exceeds image bounds", number)


def validate_dataset(layout: DatasetLayout) -> DatasetReport:
    report = DatasetReport(classes=layout.classes)
    splits = ["train", "val"] if layout.split else [""]
    digests: dict[str, list[str]] = {}
    split_digests: dict[str, set[str]] = {}
    for split in splits:
        image_dir = layout.root / "images" / split
        label_dir = layout.root / "labels" / split
        images = _images(image_dir)
        if not images:
            report.add("fatal", "missing_images", image_dir, "No images found")
        current: set[str] = set()
        for image_path in images:
            report.image_count += 1
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                report.add("fatal", "unreadable_image", image_path, str(exc))
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            digests.setdefault(digest, []).append(str(image_path))
            current.add(digest)
            label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
            if label_path.exists():
                report.label_count += 1
                _validate_label(label_path, layout.classes, report)
            else:
                report.empty_images.append(str(image_path))
        if label_dir.exists():
            for label_path in label_dir.rglob("*.txt"):
                image_base = label_path.relative_to(label_dir).with_suffix("")
                if not any((image_dir / image_base).with_suffix(ext).exists() for ext in IMAGE_SUFFIXES):
                    report.add("fatal", "missing_image", label_path, "Label has no matching image")
        split_digests[split] = current
    report.duplicate_images = [paths for paths in digests.values() if len(paths) > 1]
    for paths in report.duplicate_images:
        report.add("fatal", "duplicate_image", Path(paths[0]), f"Duplicate image: {paths[1:]}")
    if layout.split and split_digests["train"] & split_digests["val"]:
        report.add("fatal", "split_leakage", layout.root, "Train and val contain identical images")
    return report
