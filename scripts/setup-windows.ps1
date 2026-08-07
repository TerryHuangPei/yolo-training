$ErrorActionPreference = "Stop"

if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw "This bootstrap is for Windows. Use scripts/setup-macos.sh on macOS."
}

$python = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $python) {
    throw "Python 3.11 is required. Install it from https://www.python.org/downloads/windows/"
}

& py -3.11 -c "import sys" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 is required. Install it from https://www.python.org/downloads/windows/"
}

& py -3.11 -m venv .venv
New-Item -ItemType Directory -Force -Path .ultralytics, .cache\matplotlib | Out-Null
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -r requirements-windows.lock -e .
& .venv\Scripts\python.exe scripts\windows_doctor.py
