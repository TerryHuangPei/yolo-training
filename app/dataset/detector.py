from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.exceptions import DatasetFormatError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    split: bool
    classes: list[str]


def _classes_from_yaml(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def detect_dataset(root: Path) -> DatasetLayout:
    if not (root / "images").is_dir() or not (root / "labels").is_dir():
        raise DatasetFormatError(f"Expected images/ and labels/ directories: {root}")
    yaml_path = root / "dataset.yaml"
    classes_path = root / "classes.txt"
    classes = _classes_from_yaml(yaml_path) if yaml_path.exists() else []
    if not classes and classes_path.exists():
        classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise DatasetFormatError(f"Missing class names (dataset.yaml or classes.txt): {root}")
    split = (root / "images" / "train").is_dir() or (root / "images" / "val").is_dir()
    if split and not ((root / "images" / "train").is_dir() and (root / "images" / "val").is_dir()):
        raise DatasetFormatError(f"Both images/train and images/val are required: {root}")
    return DatasetLayout(root=root, split=split, classes=classes)
