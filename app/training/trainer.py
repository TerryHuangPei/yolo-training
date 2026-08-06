from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings
from app.exceptions import TrainingError


def train_model(
    dataset_yaml: Path,
    job_dir: Path,
    settings: Settings,
    model_name: str,
    epochs: int,
    imgsz: int,
    device: str,
    batch: int,
    resume: Path | None = None,
) -> tuple[Path, Path]:
    try:
        from ultralytics import YOLO

        model = YOLO(str(resume) if resume else model_name)
        run_project = (job_dir / "runs").resolve()
        result: Any = model.train(
            data=str(dataset_yaml),
            project=str(run_project),
            name="train",
            exist_ok=True,
            epochs=epochs,
            imgsz=imgsz,
            patience=settings.patience,
            workers=settings.workers,
            seed=settings.seed,
            deterministic=settings.deterministic,
            plots=settings.plots,
            save=settings.save,
            device=device,
            batch=batch,
            resume=bool(resume),
        )
        save_dir = Path(result.save_dir)
        best, last = save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"
        if not best.exists() or not last.exists():
            raise TrainingError(f"Ultralytics did not produce model weights in {save_dir}")
        return best, last
    except TrainingError:
        raise
    except Exception as exc:
        raise TrainingError(str(exc)) from exc
