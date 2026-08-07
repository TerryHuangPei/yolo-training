from __future__ import annotations

import platform
import sys

from app.training.device import select_device


def main() -> int:
    """Print the native Windows Python, PyTorch, and CUDA configuration."""
    try:
        import torch
    except ImportError:
        print(
            "PyTorch is not installed. Run: powershell -File scripts/setup-windows.ps1",
            file=sys.stderr,
        )
        return 1
    print(f"platform={platform.platform()}")
    print(f"machine={platform.machine()}")
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_built={torch.backends.cuda.is_built()}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"selected_device={select_device(None)}")
    if platform.system() != "Windows":
        print("Warning: this is not Windows.", file=sys.stderr)
    if torch.backends.cuda.is_built() and not torch.cuda.is_available():
        print(
            "Warning: PyTorch includes CUDA support, but CUDA is unavailable; "
            "the pipeline will use CPU.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
