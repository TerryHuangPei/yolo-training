#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This bootstrap is for macOS. Use Docker targets on Linux." >&2
  exit 2
fi

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 is required. Install it with: brew install python@3.11" >&2
  exit 2
fi

python3.11 -m venv .venv
mkdir -p .ultralytics .cache/matplotlib
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-macos.lock -e .
.venv/bin/python scripts/macos_doctor.py
