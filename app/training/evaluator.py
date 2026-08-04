from __future__ import annotations

from pathlib import Path
from typing import Any

from app.exceptions import EvaluationError


def evaluate_model(model_path: Path, dataset_yaml: Path, device: str, imgsz: int) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
        metrics = YOLO(str(model_path)).val(data=str(dataset_yaml), device=device, imgsz=imgsz, plots=False)
        box = metrics.box
        names = metrics.names
        per_class = {}
        for index, name in names.items():
            per_class[str(name)] = {"precision": float(box.p[index]), "recall": float(box.r[index]), "map50": float(box.ap50[index]), "map50_95": float(box.ap[index])}
        return {"precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50), "map50_95": float(box.map), "per_class": per_class, "speed_ms": {str(k): float(v) for k, v in metrics.speed.items()}, "class_names": list(names.values()), "dataset": str(dataset_yaml)}
    except Exception as exc:
        raise EvaluationError(str(exc)) from exc
