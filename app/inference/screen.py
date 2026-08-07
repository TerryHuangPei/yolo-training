from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from app.exceptions import InferenceError
from app.inference.serializer import detection_record


def measurement_records(
    frame_record: dict[str, Any],
    mouse_x: float,
    mouse_y: float,
    target_class: str,
    monitor_left: int,
    monitor_top: int,
) -> list[dict[str, Any]]:
    """Measure the pointer against every detected target class in the current frame."""
    records: list[dict[str, Any]] = []
    for detection_index, detection in enumerate(frame_record["detections"]):
        if detection["class_name"].casefold() != target_class.casefold():
            continue
        box = detection["box"]
        center_x = (box["x1"] + box["x2"]) / 2
        center_y = (box["y1"] + box["y2"]) / 2
        distance = ((mouse_x - center_x) ** 2 + (mouse_y - center_y) ** 2) ** 0.5
        records.append(
            {
                "recorded_at": datetime.now(UTC).isoformat(),
                "frame": frame_record["frame"],
                "frame_timestamp_ms": frame_record["timestamp_ms"],
                "target": {
                    "detection_index": detection_index,
                    "class_id": detection["class_id"],
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "box": box,
                },
                "mouse_position": {
                    "x": round(mouse_x, 2),
                    "y": round(mouse_y, 2),
                    "screen_x": round(monitor_left + mouse_x, 2),
                    "screen_y": round(monitor_top + mouse_y, 2),
                },
                "head_center": {
                    "x": round(center_x, 2),
                    "y": round(center_y, 2),
                    "screen_x": round(monitor_left + center_x, 2),
                    "screen_y": round(monitor_top + center_y, 2),
                },
                "distance_pixels": round(distance, 2),
            }
        )
    return records


def predict_screen(
    model_path: Path,
    conf: float,
    imgsz: int,
    device: str,
    monitor: int = 1,
    target_class: str = "head",
    output: Path | None = None,
    hotkey: str = "`",
) -> Path:
    """Run YOLO against a display and record a target measurement on a global hotkey.

    ``mss`` uses monitor 0 for the virtual desktop and starts physical displays at 1.
    """
    if len(hotkey) != 1:
        raise InferenceError("--hotkey must be exactly one character, for example '`'.")
    try:
        import cv2
        import mss
        import numpy as np
        from pynput import keyboard, mouse
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        window_name = "YOLO screen inference (global hotkey; q or Esc to stop)"
        output = output or model_path.parent.parent / "screen-clicks.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with mss.mss() as capture:
            if monitor < 1 or monitor >= len(capture.monitors):
                available = len(capture.monitors) - 1
                raise InferenceError(
                    f"Monitor {monitor} is unavailable; choose a value from 1 to {available}."
                )
            target = capture.monitors[monitor]
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            pending_hotkeys: list[tuple[int, int, str]] = []
            pending_lock = Lock()
            pointer = mouse.Controller()

            def on_press(key: object) -> None:
                if not isinstance(key, keyboard.KeyCode) or key.char != hotkey:
                    return
                mouse_x, mouse_y = pointer.position
                with pending_lock:
                    pending_hotkeys.append(
                        (round(mouse_x), round(mouse_y), datetime.now(UTC).isoformat())
                    )

            listener = keyboard.Listener(on_press=on_press)
            listener.start()
            started_at = time.perf_counter()
            frame_no = 0
            try:
                with output.open("a", encoding="utf-8") as measurements:
                    while True:
                        started = time.perf_counter()
                        # MSS yields BGRA; YOLO/OpenCV use BGR images.
                        frame = np.asarray(capture.grab(target))[:, :, :3]
                        frame_no += 1
                        result = model(
                            frame, conf=conf, imgsz=imgsz, device=device, verbose=False
                        )[0]
                        frame_record = detection_record(
                            frame_no,
                            round((time.perf_counter() - started_at) * 1000),
                            result.boxes,
                            result.names,
                        )
                        with pending_lock:
                            presses = pending_hotkeys.copy()
                            pending_hotkeys.clear()
                        for mouse_x, mouse_y, triggered_at in presses:
                            # The pointer location is relative to the physical desktop;
                            # convert it to the captured monitor image used by YOLO.
                            click_x = mouse_x - target["left"]
                            click_y = mouse_y - target["top"]
                            measurements_for_press = measurement_records(
                                frame_record,
                                click_x,
                                click_y,
                                target_class,
                                target["left"],
                                target["top"],
                            )
                            if measurements_for_press:
                                for measurement in measurements_for_press:
                                    measurement["triggered_at"] = triggered_at
                                    measurements.write(
                                        json.dumps(measurement, ensure_ascii=False) + "\n"
                                    )
                                measurements.flush()
                                print(
                                    f"Recorded {len(measurements_for_press)} "
                                    f"{target_class} measurement(s) "
                                    f"from global hotkey in {output}"
                                )
                            else:
                                print(
                                    f"No '{target_class}' boxes in the current frame; "
                                    "nothing recorded."
                                )
                        annotated = result.plot()
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        cv2.putText(
                            annotated,
                            f"Press {hotkey}: measure pointer to all {target_class} boxes | "
                            f"{elapsed_ms:.0f} ms/frame | q/Esc: stop",
                            (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(window_name, annotated)
                        if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                            break
            finally:
                listener.stop()
                listener.join()
                cv2.destroyWindow(window_name)
        return output
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(
            "Unable to capture the screen. On macOS, grant Screen Recording permission to "
            f"the application running this command. Details: {exc}"
        ) from exc
