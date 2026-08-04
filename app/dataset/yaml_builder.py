from __future__ import annotations

from pathlib import Path

import yaml


def build_dataset_yaml(root: Path, classes: list[str]) -> Path:
    path = root / "dataset.yaml"
    payload = {"path": str(root), "train": "images/train", "val": "images/val", "names": classes, "nc": len(classes)}
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
