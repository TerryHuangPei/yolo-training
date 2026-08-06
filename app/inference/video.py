from __future__ import annotations

import json
import time
from pathlib import Path

from app.exceptions import InferenceError
from app.inference.serializer import detection_record
from app.inference.summary import InferenceSummary


def predict_video(
    model_path: Path, source: Path, output_dir: Path, conf: float, imgsz: int, device: str
) -> dict[str, object]:
    try:
        import cv2
        from ultralytics import YOLO

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise InferenceError(f"Cannot open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width, height = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path, jsonl_path = output_dir / "annotated-video.mp4", output_dir / "detections.jsonl"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise InferenceError(f"Cannot create output video: {video_path}")
        model, summary, frame_no = (
            YOLO(str(model_path)),
            InferenceSummary(fps, total, str(video_path)),
            0,
        )
        try:
            with jsonl_path.open("w", encoding="utf-8") as handle:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_no += 1
                    started = time.perf_counter()
                    result = model(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
                    record = detection_record(
                        frame_no, round((frame_no - 1) * 1000 / fps), result.boxes, result.names
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    summary.add(record["detections"], (time.perf_counter() - started) * 1000)
                    writer.write(result.plot())
        finally:
            capture.release()
            writer.release()
        payload = summary.payload()
        (output_dir / "summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return payload
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(str(exc)) from exc
