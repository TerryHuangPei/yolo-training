from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.pipeline.state import JobStatus, validate_transition


def now() -> str:
    return datetime.now(UTC).isoformat()


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path

    def create(self, job_id: str, project_name: str) -> dict[str, Any]:
        data: dict[str, Any] = {"job_id": job_id, "project_name": project_name, "status": JobStatus.CREATED, "created_at": now(), "updated_at": now(), "warnings": []}
        self._write(data)
        return data

    def read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def transition(self, status: JobStatus, **changes: Any) -> dict[str, Any]:
        data = self.read()
        validate_transition(JobStatus(data["status"]), status)
        data.update(changes, status=status, updated_at=now())
        self._write(data)
        return data

    def fail(self, stage: str, exc: BaseException, trace: str) -> None:
        data = self.read()
        data.update(status=JobStatus.FAILED, failed_stage=stage, exception_type=type(exc).__name__, message=str(exc), traceback=trace, updated_at=now())
        self._write(data)

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(temporary, self.path)
