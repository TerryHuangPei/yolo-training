from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.dataset.detector import detect_dataset
from app.dataset.extractor import extract_dataset
from app.dataset.splitter import split_dataset
from app.dataset.validator import validate_dataset
from app.dataset.yaml_builder import build_dataset_yaml
from app.exceptions import DatasetExtractionError


def make_dataset(root: Path, split: bool = False, label: str = "0 0.5 0.5 0.2 0.2") -> Path:
    section = "train" if split else ""
    image_dir, label_dir = root / "images" / section, root / "labels" / section
    image_dir.mkdir(parents=True); label_dir.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(image_dir / "a.jpg")
    (label_dir / "a.txt").write_text(label, encoding="utf-8")
    (root / "classes.txt").write_text("helmet\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("label", ["0 0.5 0.5 0.2", "1 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.95 0.5 0.2 0.2"])
def test_invalid_labels(tmp_path: Path, label: str) -> None:
    report = validate_dataset(detect_dataset(make_dataset(tmp_path, label=label)))
    assert report.fatal_count == 1


def test_missing_label_is_empty_image(tmp_path: Path) -> None:
    root = make_dataset(tmp_path); (root / "labels" / "a.txt").unlink()
    report = validate_dataset(detect_dataset(root))
    assert report.fatal_count == 0 and report.empty_images


def test_duplicate_and_leakage(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, split=True)
    (root / "images" / "val").mkdir(); (root / "labels" / "val").mkdir()
    (root / "images" / "val" / "a.jpg").write_bytes((root / "images" / "train" / "a.jpg").read_bytes())
    (root / "labels" / "val" / "a.txt").write_text("0 0.5 0.5 0.2 0.2")
    report = validate_dataset(detect_dataset(root))
    assert any(issue.code == "split_leakage" for issue in report.issues)


def test_split_is_reproducible_and_yaml(tmp_path: Path) -> None:
    root = make_dataset(tmp_path)
    for index in range(1, 5):
        Image.new("RGB", (10, 10)).save(root / "images" / f"{index}.jpg")
        (root / "labels" / f"{index}.txt").write_text("0 0.5 0.5 0.2 0.2")
    split_dataset(root, 0.8, 42)
    assert len(list((root / "images" / "train").glob("*.jpg"))) == 4
    assert "images/train" in build_dataset_yaml(root, ["helmet"]).read_text()


def test_zip_slip(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle: handle.writestr("../bad", "x")
    with pytest.raises(DatasetExtractionError): extract_dataset(archive, tmp_path / "out", Settings())


def test_zip_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "link.zip"
    info = zipfile.ZipInfo("link")
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as handle: handle.writestr(info, "target")
    with pytest.raises(DatasetExtractionError): extract_dataset(archive, tmp_path / "out", Settings())
