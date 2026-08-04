from __future__ import annotations

import random
import shutil
from pathlib import Path

from app.dataset.detector import IMAGE_SUFFIXES
from app.exceptions import DatasetFormatError


def split_dataset(root: Path, train_ratio: float, seed: int) -> None:
    """Create reproducible train/val copies within an extracted job dataset."""
    images_dir, labels_dir = root / "images", root / "labels"
    images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < 2:
        raise DatasetFormatError("At least two images are required to split a dataset")
    random.Random(seed).shuffle(images)
    boundary = max(1, min(len(images) - 1, int(len(images) * train_ratio)))
    selected = {"train": images[:boundary], "val": images[boundary:]}
    staged = root / ".split-staging"
    (staged / "images").mkdir(parents=True)
    (staged / "labels").mkdir(parents=True)
    for split, group in selected.items():
        for image in group:
            relative = image.relative_to(images_dir)
            target = staged / "images" / split / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
            label = labels_dir / relative.with_suffix(".txt")
            if label.exists():
                label_target = staged / "labels" / split / relative.with_suffix(".txt")
                label_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label, label_target)
    shutil.rmtree(images_dir)
    shutil.rmtree(labels_dir)
    shutil.move(str(staged / "images"), str(images_dir))
    shutil.move(str(staged / "labels"), str(labels_dir))
    staged.rmdir()
