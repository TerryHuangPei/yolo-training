from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.exceptions import ArtifactPromotionError


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".tmp")
        with source.open("rb") as read_handle, temp.open("wb") as write_handle:
            shutil.copyfileobj(read_handle, write_handle)
            write_handle.flush()
            os.fsync(write_handle.fileno())
        os.replace(temp, destination)
    except Exception as exc:
        raise ArtifactPromotionError(f"Could not promote {source}: {exc}") from exc


def promote_artifacts(
    best_source: Path,
    last_source: Path,
    model_dir: Path,
    job_id: str,
    project_name: str,
    base_model: str,
    classes: list[str],
    dataset_stats: dict[str, Any],
    training_parameters: dict[str, Any],
    validation_metrics: dict[str, Any],
) -> Path:
    best, last = model_dir / "best.pt", model_dir / "last.pt"
    atomic_copy(best_source, best)
    atomic_copy(last_source, last)
    checksum = sha256(best)
    metadata = {
        "job_id": job_id,
        "project_name": project_name,
        "task": "detect",
        "base_model": base_model,
        "ultralytics_version": "8.4.115",
        "created_at": datetime.now(UTC).isoformat(),
        "classes": classes,
        "dataset_stats": dataset_stats,
        "training_parameters": training_parameters,
        "validation_metrics": validation_metrics,
        "model_filename": "best.pt",
        "model_size_bytes": best.stat().st_size,
        "sha256": checksum,
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (model_dir / "sha256.txt").write_text(f"{checksum}  best.pt\n", encoding="utf-8")
    return best
