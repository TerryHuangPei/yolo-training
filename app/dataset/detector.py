from __future__ import annotations

import shutil
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
    _normalise_roboflow_layout(root)
    if not (root / "images").is_dir() or not (root / "labels").is_dir():
        raise DatasetFormatError(f"Expected images/ and labels/ directories: {root}")
    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists() and (root / "data.yaml").exists():
        yaml_path = root / "data.yaml"
    classes_path = root / "classes.txt"
    classes = _classes_from_yaml(yaml_path) if yaml_path.exists() else []
    if not classes and classes_path.exists():
        classes = [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not classes:
        raise DatasetFormatError(f"Missing class names (dataset.yaml or classes.txt): {root}")
    split = (root / "images" / "train").is_dir() or (root / "images" / "val").is_dir()
    if split and not ((root / "images" / "train").is_dir() and (root / "images" / "val").is_dir()):
        raise DatasetFormatError(f"Both images/train and images/val are required: {root}")
    return DatasetLayout(root=root, split=split, classes=classes)


def _normalise_roboflow_layout(root: Path) -> None:
    """Normalize Roboflow's train/images and valid/images layout in a job copy."""
    train_images = root / "train" / "images"
    valid_images = root / "valid" / "images"
    if not train_images.is_dir() and not valid_images.is_dir():
        return
    if not train_images.is_dir() or not valid_images.is_dir():
        raise DatasetFormatError(
            f"Roboflow dataset requires both train/images and valid/images: {root}"
        )
    images_dir = root / "images"
    labels_dir = root / "labels"
    if images_dir.exists() or labels_dir.exists():
        raise DatasetFormatError(f"Ambiguous mixed dataset layouts: {root}")
    for source_name, target_name in (("train", "train"), ("valid", "val")):
        image_source = root / source_name / "images"
        label_source = root / source_name / "labels"
        if not label_source.is_dir():
            raise DatasetFormatError(f"Missing Roboflow label directory: {label_source}")
        (images_dir).mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        shutil.move(str(image_source), str(images_dir / target_name))
        shutil.move(str(label_source), str(labels_dir / target_name))
