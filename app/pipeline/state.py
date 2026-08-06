from __future__ import annotations

from enum import StrEnum

from app.exceptions import JobStateError


class JobStatus(StrEnum):
    CREATED = "created"
    VALIDATING = "validating"
    INVALID = "invalid"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    INFERENCING = "inferencing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.CREATED: {JobStatus.VALIDATING, JobStatus.FAILED},
    JobStatus.VALIDATING: {JobStatus.INVALID, JobStatus.PREPARING, JobStatus.FAILED},
    JobStatus.INVALID: set(),
    JobStatus.PREPARING: {JobStatus.TRAINING, JobStatus.FAILED},
    JobStatus.TRAINING: {JobStatus.EVALUATING, JobStatus.FAILED, JobStatus.INTERRUPTED},
    JobStatus.EVALUATING: {JobStatus.INFERENCING, JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.INFERENCING: {JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.INTERRUPTED: set(),
}


def validate_transition(current: JobStatus, target: JobStatus) -> None:
    if target not in TRANSITIONS[current]:
        raise JobStateError(f"Invalid state transition: {current} -> {target}")
