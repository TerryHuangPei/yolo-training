from __future__ import annotations


def select_device(explicit: str | None, configured: str | None = None) -> str:
    if explicit:
        return explicit
    if configured:
        return configured
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def select_batch(explicit: int | None, device: str) -> int:
    return explicit if explicit is not None else (-1 if device not in {"cpu", "mps"} else 8)
