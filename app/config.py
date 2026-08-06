from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YOLO_", env_file=".env", extra="ignore")
    input_dir: Path = Path("input")
    jobs_dir: Path = Path("workspace")
    artifacts_dir: Path = Path("artifacts")
    model: str = "yolo26n.pt"
    epochs: int = 100
    imgsz: int = 640
    patience: int = 30
    workers: int = 4
    seed: int = 42
    deterministic: bool = True
    plots: bool = True
    save: bool = True
    device: str | None = None
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    max_zip_bytes: int = Field(default=5 * 1024 * 1024 * 1024, gt=0)
    max_zip_files: int = Field(default=100_000, gt=0)
    max_compression_ratio: float = Field(default=100.0, gt=1)


def load_yaml_settings(path: Path | None) -> dict[str, object]:
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def settings_from_yaml(path: Path | None = None) -> Settings:
    return Settings(**load_yaml_settings(path))
