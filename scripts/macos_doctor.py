from __future__ import annotations

import platform
import sys

from app.training.device import select_device


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Run: make mac-setup", file=sys.stderr)
        return 1
    print(f"platform={platform.platform()}")
    print(f"machine={platform.machine()}")
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"mps_built={torch.backends.mps.is_built()}")
    print(f"mps_available={torch.backends.mps.is_available()}")
    print(f"selected_device={select_device(None)}")
    if torch.backends.mps.is_built() and not torch.backends.mps.is_available():
        print(
            "Warning: PyTorch includes MPS support, but the current process cannot access it; "
            "the pipeline will use CPU.",
            file=sys.stderr,
        )
    if platform.system() != "Darwin":
        print("Warning: this is not macOS.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
