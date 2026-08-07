from __future__ import annotations

from pathlib import Path

import pytest

from app.exceptions import JobStateError
from app.inference.screen import draw_mouse_cursor, measurement_records, move_to_nearest_head
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


def test_move_to_nearest_head_uses_recorded_screen_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurements = [
        {
            "distance_pixels": 20.0,
            "mouse_position": {"screen_x": 10.0, "screen_y": 20.0},
            "head_center": {"screen_x": 30.0, "screen_y": 20.0},
        },
        {
            "distance_pixels": 5.0,
            "mouse_position": {"screen_x": 10.0, "screen_y": 20.0},
            "head_center": {"screen_x": 13.0, "screen_y": 24.0},
        },
    ]
    calls: list[dict[str, object]] = []

    def fake_move_mouse(**kwargs: object) -> tuple[float, float]:
        calls.append(kwargs)
        return 3.0, 4.0

    monkeypatch.setattr("app.inference.screen.move_mouse", fake_move_mouse)
    selected = move_to_nearest_head(measurements, 90, 1.5, 0, 2, True)
    assert selected is measurements[1]
    assert calls == [
        {
            "point_a": (10.0, 20.0),
            "point_b": (13.0, 24.0),
            "fov": 90,
            "sensitivity": 1.5,
            "duration": 0,
            "steps": 2,
            "smooth": True,
        }
    ]
    assert selected["mouse_movement"]["actual_dx"] == 3.0


def test_draw_mouse_cursor_only_draws_when_pointer_is_on_captured_monitor() -> None:
    class FakeCv2:
        MARKER_CROSS = 1
        LINE_AA = 2
        FONT_HERSHEY_SIMPLEX = 3

        def __init__(self) -> None:
            self.calls: list[str] = []

        def drawMarker(self, *args: object, **kwargs: object) -> None:  # noqa: N802
            self.calls.append("marker")

        def circle(self, *args: object, **kwargs: object) -> None:
            self.calls.append("circle")

        def putText(self, *args: object, **kwargs: object) -> None:  # noqa: N802
            self.calls.append("text")

    class FakeImage:
        shape = (100, 200, 3)

    cv2 = FakeCv2()
    image = FakeImage()
    assert draw_mouse_cursor(image, 110, 220, 100, 200, cv2)
    assert cv2.calls == ["marker", "circle", "text"]
    assert not draw_mouse_cursor(image, 99, 220, 100, 200, cv2)
