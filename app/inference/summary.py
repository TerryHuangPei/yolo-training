from __future__ import annotations

from collections import defaultdict
from typing import Any


class InferenceSummary:
    def __init__(self, fps: float, total_frames: int, output_video: str) -> None:
        self.fps, self.total_frames, self.output_video = fps, total_frames, output_video
        self.processed_frames = self.total_detections = 0
        self.counts: dict[str, int] = defaultdict(int)
        self.confidences: dict[str, list[float]] = defaultdict(list)
        self.times: list[float] = []

    def add(self, detections: list[dict[str, Any]], inference_ms: float) -> None:
        self.processed_frames += 1
        self.times.append(inference_ms)
        self.total_detections += len(detections)
        for detection in detections:
            name = str(detection["class_name"])
            self.counts[name] += 1
            self.confidences[name].append(float(detection["confidence"]))

    def payload(self) -> dict[str, Any]:
        return {"total_frames": self.total_frames, "processed_frames": self.processed_frames, "duration_seconds": self.total_frames / self.fps if self.fps else 0, "source_fps": self.fps, "output_fps": self.fps, "total_detections": self.total_detections, "detections_by_class": dict(self.counts), "average_confidence_by_class": {key: sum(values) / len(values) for key, values in self.confidences.items()}, "inference_ms_average": sum(self.times) / len(self.times) if self.times else 0, "output_video": self.output_video}
