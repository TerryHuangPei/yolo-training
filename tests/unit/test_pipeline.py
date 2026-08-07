from __future__ import annotations

from pathlib import Path

import pytest

from app.exceptions import JobStateError
from app.inference.screen import measurement_records
from app.inference.serializer import detection_record
from app.pipeline.manifest import Manifest
from app.pipeline.state import JobStatus
from app.training.artifacts import sha256


def test_sha256(tmp_path: Path) -> None:
    path = tmp_path / "data"
    path.write_text("abc")
    assert sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_manifest_transitions(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.create("id", "name")
    manifest.transition(JobStatus.VALIDATING)
    assert manifest.read()["status"] == "validating"
    with pytest.raises(JobStateError):
        manifest.transition(JobStatus.COMPLETED)


class Value:
    def __init__(self, value: object) -> None:
        self.value = value

    def item(self) -> object:
        return self.value

    def tolist(self) -> object:
        return self.value


class Box:
    cls = Value(0)
    conf = Value(0.9)
    xyxy = [Value([1, 2, 3, 4])]


def test_jsonl_serializer() -> None:
    record = detection_record(1, 0, [Box()], {0: "helmet"})
    assert record["detections"][0]["class_name"] == "helmet"


def test_measurement_records_all_heads_even_when_pointer_is_outside_boxes() -> None:
    frame = {
        "frame": 4,
        "timestamp_ms": 120,
        "detections": [
            {
                "class_id": 0,
                "class_name": "head",
                "confidence": 0.9,
                "box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
            },
            {
                "class_id": 0,
                "class_name": "head",
                "confidence": 0.8,
                "box": {"x1": 50, "y1": 60, "x2": 70, "y2": 80},
            },
        ],
    }
    records = measurement_records(frame, 0, 0, "head", 100, 200)
    assert len(records) == 2
    record = records[0]
    assert record["mouse_position"] == {"x": 0, "y": 0, "screen_x": 100, "screen_y": 200}
    assert record["head_center"] == {"x": 20, "y": 30, "screen_x": 120, "screen_y": 230}
    assert record["distance_pixels"] == 36.06
