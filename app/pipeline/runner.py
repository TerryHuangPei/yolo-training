from __future__ import annotations

import json
import re
import secrets
import shutil
import traceback
from datetime import datetime
from pathlib import Path

from app.config import Settings
from app.dataset.detector import detect_dataset
from app.dataset.extractor import extract_dataset
from app.dataset.splitter import split_dataset
from app.dataset.validator import validate_dataset
from app.dataset.yaml_builder import build_dataset_yaml
from app.exceptions import DatasetValidationError
from app.inference.video import predict_video
from app.logging_config import configure_logging, log_event
from app.pipeline.manifest import Manifest
from app.pipeline.state import JobStatus
from app.training.artifacts import promote_artifacts
from app.training.device import select_batch, select_device
from app.training.evaluator import evaluate_model
from app.training.trainer import train_model


def make_job_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"
    return f"{datetime.now():%Y%m%d-%H%M%S}-{slug}-{secrets.token_hex(3)}"


class PipelineRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _create(self, name: str) -> tuple[str, Path, Manifest]:
        job_id = make_job_id(name)
        artifact = self.settings.artifacts_dir / "jobs" / job_id
        work = self.settings.jobs_dir / job_id
        work.mkdir(parents=True, exist_ok=False)
        artifact.mkdir(parents=True, exist_ok=False)
        (artifact / "logs").mkdir()
        (artifact / "logs" / "pipeline.log").write_text("", encoding="utf-8")
        log_event(
            configure_logging(artifact / "logs" / "pipeline.log"), "job_created", job_id=job_id
        )
        manifest = Manifest(artifact / "manifest.json")
        manifest.create(job_id, name)
        return job_id, work, manifest

    def prepare(
        self, dataset: Path, name: str
    ) -> tuple[str, Path, Path, list[str], Manifest, dict[str, int]]:
        job_id, work, manifest = self._create(name)
        try:
            manifest.transition(JobStatus.VALIDATING)
            root = extract_dataset(dataset, work / "dataset", self.settings)
            layout = detect_dataset(root)
            report = validate_dataset(layout)
            report.write(manifest.path.parent / "reports")
            if report.fatal_count:
                manifest.transition(JobStatus.INVALID, fatal_errors=report.fatal_count)
                raise DatasetValidationError(
                    f"Dataset validation failed with {report.fatal_count} fatal error(s)"
                )
            manifest.transition(JobStatus.PREPARING)
            if not layout.split:
                split_dataset(root, self.settings.train_ratio, self.settings.seed)
            yaml_path = build_dataset_yaml(root, layout.classes)
            return (
                job_id,
                work,
                yaml_path,
                layout.classes,
                manifest,
                {
                    "images": report.image_count,
                    "labels": report.label_count,
                    "empty_images": len(report.empty_images),
                },
            )
        except Exception as exc:
            (manifest.path.parent / "logs" / "pipeline.log").open("a", encoding="utf-8").write(
                traceback.format_exc()
            )
            if manifest.read()["status"] not in {JobStatus.INVALID, JobStatus.FAILED}:
                manifest.fail("preparing", exc, traceback.format_exc())
            raise

    def pipeline(
        self,
        dataset: Path,
        name: str,
        model: str,
        epochs: int,
        imgsz: int,
        video: Path | None = None,
        device: str | None = None,
        batch: int | None = None,
        min_map50: float | None = None,
        min_map50_95: float | None = None,
        resume: Path | None = None,
    ) -> str:
        job_id, work, yaml_path, classes, manifest, stats = self.prepare(dataset, name)
        chosen_device = select_device(device, self.settings.device)
        chosen_batch = select_batch(batch, chosen_device)
        try:
            manifest.transition(JobStatus.TRAINING)
            best_source, last_source = train_model(
                yaml_path,
                work,
                self.settings,
                model,
                epochs,
                imgsz,
                chosen_device,
                chosen_batch,
                resume,
            )
            manifest.transition(JobStatus.EVALUATING)
            metrics = evaluate_model(best_source, yaml_path, chosen_device, imgsz)
            reports = manifest.path.parent / "reports"
            reports.mkdir(exist_ok=True)
            (reports / "validation-metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            results = best_source.parent.parent / "results.csv"
            if results.exists():
                shutil.copy2(results, reports / "results.csv")
            warnings: list[str] = []
            if min_map50 is not None and metrics["map50"] < min_map50:
                raise DatasetValidationError("map50 promotion threshold was not met")
            if min_map50_95 is not None and metrics["map50_95"] < min_map50_95:
                raise DatasetValidationError("map50_95 promotion threshold was not met")
            if (
                min_map50 is None
                and min_map50_95 is None
                and (metrics["map50"] < 0.5 or metrics["map50_95"] < 0.25)
            ):
                warnings.append(
                    "Validation metrics are low; model promoted because no threshold was requested."
                )
            best = promote_artifacts(
                best_source,
                last_source,
                manifest.path.parent / "model",
                job_id,
                name,
                model,
                classes,
                stats,
                {"epochs": epochs, "imgsz": imgsz, "device": chosen_device, "batch": chosen_batch},
                metrics,
            )
            (reports / "training-summary.json").write_text(
                json.dumps(
                    {"job_id": job_id, "best_model": str(best), "metrics": metrics}, indent=2
                ),
                encoding="utf-8",
            )
            if video:
                manifest.transition(JobStatus.INFERENCING, warnings=warnings)
                predict_video(
                    best, video, manifest.path.parent / "inference", 0.25, imgsz, chosen_device
                )
            manifest.transition(JobStatus.COMPLETED, warnings=warnings)
            return job_id
        except KeyboardInterrupt as exc:
            manifest.transition(JobStatus.INTERRUPTED)
            raise exc
        except Exception as exc:
            (manifest.path.parent / "logs" / "pipeline.log").open("a", encoding="utf-8").write(
                traceback.format_exc()
            )
            manifest.fail("training", exc, traceback.format_exc())
            raise
