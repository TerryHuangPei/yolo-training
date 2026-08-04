from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from app.config import Settings
from app.exceptions import DatasetExtractionError


def _safe_member(member: zipfile.ZipInfo, destination: Path) -> Path:
    name = member.filename.replace("\\", "/")
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts or name.startswith("/"):
        raise DatasetExtractionError(f"Unsafe ZIP path: {member.filename}")
    if member.is_dir():
        return destination / candidate
    mode = member.external_attr >> 16
    if mode and (mode & 0o170000) == 0o120000:
        raise DatasetExtractionError(f"Symlink is forbidden in ZIP: {member.filename}")
    return destination / candidate


def extract_dataset(source: Path, destination: Path, settings: Settings) -> Path:
    """Copy a directory or safely extract a ZIP into a job-owned destination."""
    if not source.exists():
        raise DatasetExtractionError(f"Dataset does not exist: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=False)
        return destination
    if source.suffix.lower() != ".zip":
        raise DatasetExtractionError(f"Dataset must be a directory or ZIP: {source}")
    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            if len(infos) > settings.max_zip_files:
                raise DatasetExtractionError(f"ZIP has too many files: {len(infos)}")
            total_size = sum(info.file_size for info in infos)
            if total_size > settings.max_zip_bytes:
                raise DatasetExtractionError(f"ZIP extracted size exceeds limit: {total_size}")
            for info in infos:
                if info.compress_size and info.file_size / info.compress_size > settings.max_compression_ratio:
                    raise DatasetExtractionError(f"Suspicious compression ratio: {info.filename}")
                target = _safe_member(info, destination)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile as exc:
        raise DatasetExtractionError(f"Invalid ZIP: {source}") from exc
    return destination
