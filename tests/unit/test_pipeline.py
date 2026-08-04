from __future__ import annotations

from pathlib import Path

import pytest

from app.inference.serializer import detection_record
from app.pipeline.manifest import Manifest
from app.pipeline.state import JobStatus
from app.training.artifacts import sha256


def test_sha256(tmp_path: Path) -> None:
    path = tmp_path / "data"; path.write_text("abc")
    assert sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_manifest_transitions(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.json"); manifest.create("id", "name")
    manifest.transition(JobStatus.VALIDATING)
    assert manifest.read()["status"] == "validating"
    with pytest.raises(Exception): manifest.transition(JobStatus.COMPLETED)


class Value:
    def __init__(self, value: object) -> None: self.value = value
    def item(self) -> object: return self.value
    def tolist(self) -> object: return self.value


class Box:
    cls = Value(0); conf = Value(0.9); xyxy = [Value([1, 2, 3, 4])]


def test_jsonl_serializer() -> None:
    record = detection_record(1, 0, [Box()], {0: "helmet"})
    assert record["detections"][0]["class_name"] == "helmet"
