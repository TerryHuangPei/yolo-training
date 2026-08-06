from __future__ import annotations

import time
from pathlib import Path

from app.exceptions import InferenceError


def predict_screen(
    model_path: Path,
    conf: float,
    imgsz: int,
    device: str,
    monitor: int = 1,
) -> None:
    """Run YOLO against a physical display until the user presses q or Escape.

    ``mss`` uses monitor 0 for the virtual desktop and starts physical displays at 1.
    """
    try:
        import cv2
        import mss
        import numpy as np
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        window_name = "YOLO screen inference (q or Esc to stop)"
        with mss.mss() as capture:
            if monitor < 1 or monitor >= len(capture.monitors):
                available = len(capture.monitors) - 1
                raise InferenceError(
                    f"Monitor {monitor} is unavailable; choose a value from 1 to {available}."
                )
            target = capture.monitors[monitor]
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            try:
                while True:
                    started = time.perf_counter()
                    # MSS yields BGRA; YOLO/OpenCV use BGR images.
                    frame = np.asarray(capture.grab(target))[:, :, :3]
                    result = model(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
                    annotated = result.plot()
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    cv2.putText(
                        annotated,
                        f"{elapsed_ms:.0f} ms/frame | q or Esc: stop",
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                        break
            finally:
                cv2.destroyWindow(window_name)
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(
            "Unable to capture the screen. On macOS, grant Screen Recording permission to "
            f"the application running this command. Details: {exc}"
        ) from exc
