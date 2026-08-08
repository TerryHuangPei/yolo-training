from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer

from app.config import settings_from_yaml
from app.exceptions import PipelineError
from app.inference.screen import predict_screen
from app.inference.video import predict_video
from app.pipeline.manifest import Manifest
from app.pipeline.runner import PipelineRunner
from app.training.device import select_device

app = typer.Typer(no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")


def handle(fn: object) -> object:
    @wraps(fn)  # type: ignore[arg-type]
    def wrapped(*args: object, **kwargs: object) -> None:
        try:
            fn(*args, **kwargs)
        except (PipelineError, OSError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from None

    return wrapped


@app.command()
@handle
def pipeline(
    dataset: Annotated[Path, typer.Option("--dataset")],
    name: Annotated[str, typer.Option("--name")],
    video: Path | None = None,
    model: str = "yolo26n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    device: str | None = None,
    batch: int | None = None,
    min_map50: float | None = None,
    min_map50_95: float | None = None,
) -> None:
    job_id = PipelineRunner(settings_from_yaml()).pipeline(
        dataset, name, model, epochs, imgsz, video, device, batch, min_map50, min_map50_95
    )
    typer.echo(job_id)


@dataset_app.command("validate")
@handle
def validate(dataset: Annotated[Path, typer.Option("--dataset")]) -> None:
    job_id, *_ = PipelineRunner(settings_from_yaml()).prepare(dataset, "dataset-validation")
    typer.echo(job_id)


@app.command()
@handle
def train(
    dataset: Annotated[Path | None, typer.Option("--dataset")] = None,
    name: Annotated[str, typer.Option("--name")] = "training",
    model: str = "yolo26n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    resume: Path | None = None,
    device: str | None = None,
    batch: int | None = None,
) -> None:
    settings = settings_from_yaml()
    if resume and dataset is None:
        dataset = settings.jobs_dir / resume.parent.parent.name / "dataset"
    if dataset is None:
        raise typer.BadParameter("--dataset is required")
    typer.echo(
        PipelineRunner(settings).pipeline(
            dataset, name, model, epochs, imgsz, device=device, batch=batch, resume=resume
        )
    )


@app.command("predict-video")
@handle
def predict_video_command(
    model: Path, source: Path, conf: float = 0.25, imgsz: int = 640, device: str | None = None
) -> None:
    output = (
        model.parent.parent / "inference"
        if model.name == "best.pt"
        else settings_from_yaml().artifacts_dir / "inference"
    )
    typer.echo(json.dumps(predict_video(model, source, output, conf, imgsz, select_device(device))))


@app.command("predict-screen")
@handle
def predict_screen_command(
    model: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    monitor: int = 1,
    target_class: str = "head",
    output: Path | None = None,
    hotkey: str = "`",
    move_to_head: bool = False,
    move_fov: float = 90.0,
    move_sensitivity: float = 1.0,
    move_duration: float = 0.2,
    move_steps: int = 20,
    move_smooth: bool = True,
    trigger: str = "keyboard",
    window_title: str | None = None,
) -> None:
    """Capture a screen/window and persist measurements; q/Esc stops."""
    typer.echo(
        predict_screen(
            model,
            conf,
            imgsz,
            select_device(device),
            monitor,
            target_class,
            output,
            hotkey,
            move_to_head,
            move_fov,
            move_sensitivity,
            move_duration,
            move_steps,
            move_smooth,
            trigger,
            window_title,
        )
    )


@app.command("job")
def job_show(
    command: Annotated[str, typer.Argument()], job_id: Annotated[str, typer.Option("--job-id")]
) -> None:
    if command != "show":
        raise typer.BadParameter("Only 'show' is supported")
    typer.echo(
        json.dumps(
            Manifest(settings_from_yaml().artifacts_dir / "jobs" / job_id / "manifest.json").read(),
            indent=2,
        )
    )
