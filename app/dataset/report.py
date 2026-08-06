from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Issue:
    severity: str
    code: str
    path: str
    message: str
    line: int | None = None


@dataclass
class DatasetReport:
    classes: list[str]
    image_count: int = 0
    label_count: int = 0
    empty_images: list[str] = field(default_factory=list)
    duplicate_images: list[list[str]] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def fatal_count(self) -> int:
        return sum(issue.severity == "fatal" for issue in self.issues)

    def add(
        self, severity: str, code: str, path: Path, message: str, line: int | None = None
    ) -> None:
        self.issues.append(Issue(severity, code, str(path), message, line))

    def write(self, reports_dir: Path) -> None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        payload = asdict(self) | {"fatal_count": self.fatal_count}
        (reports_dir / "dataset-report.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        lines = [
            f"Images: {self.image_count}",
            f"Labels: {self.label_count}",
            f"Fatal errors: {self.fatal_count}",
        ]
        for item in self.issues:
            line_suffix = f":{item.line}" if item.line else ""
            lines.append(f"[{item.severity}] {item.path}{line_suffix} {item.message}")
        (reports_dir / "dataset-report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
