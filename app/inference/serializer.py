from __future__ import annotations

from typing import Any


def detection_record(
    frame: int, timestamp_ms: int, boxes: Any, names: dict[int, str]
) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []
    for box in boxes:
        class_id = int(box.cls.item())
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        detections.append(
            {
                "class_id": class_id,
                "class_name": str(names[class_id]),
                "confidence": float(box.conf.item()),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )
    return {"frame": frame, "timestamp_ms": timestamp_ms, "detections": detections}
