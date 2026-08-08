# YOLO Training Pipeline

Automation for validating YOLO detection datasets, training Ultralytics YOLO, promoting `best.pt`, and streaming video inference. It supports Docker on Linux/NVIDIA plus native macOS and Windows execution.

## Input datasets

Split datasets contain `images/train`, `images/val`, parallel `labels/train`, `labels/val`, and `dataset.yaml`. Unsplit datasets contain `images`, `labels`, and `classes.txt`; they are copied to the job workspace and split 80/20 with seed 42. Labels use `class_id x_center y_center width height`, normalized to 0–1. Empty images may omit a label.

Validation detects corrupt images, malformed labels, invalid classes/boxes, missing counterparts, exact duplicate images, split leakage, Zip Slip, ZIP symlinks, limits, and compression bombs. Reports are written to each job.

## Run

### Native macOS (Apple Silicon)

Native macOS deliberately does not use the Linux `amd64` Ultralytics image. This avoids Docker's `linux/amd64` versus `linux/arm64/v8` platform mismatch. Python 3.11 is required; install it with Homebrew if needed:

```sh
brew install python@3.11
make mac-setup
make mac-doctor
make mac-lint
make mac-test
```

`mac-doctor` reports whether Metal Performance Shaders (MPS) is available. The pipeline automatically selects `mps` when it is available, otherwise CPU. To train, put the archive in `input/dataset.zip` and run:

```sh
make mac-pipeline
# Or pass paths and settings explicitly:
.venv/bin/yolo-pipeline pipeline --dataset input/dataset.zip --name helmet-v1 --epochs 100 --imgsz 640
```

For native video inference:

```sh
.venv/bin/yolo-pipeline predict-video --model artifacts/jobs/JOB_ID/model/best.pt --source input/test.mp4
```

Native macOS jobs use the repository-local `input/`, `workspace/`, and `artifacts/` directories. Docker continues to use its `/workspace/...` mounts via environment variables.

### Native Windows

Windows requires Python 3.11. In PowerShell, create the virtual environment and install the native dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-windows.ps1
.venv\Scripts\python.exe scripts\windows_doctor.py
.venv\Scripts\pytest.exe
.venv\Scripts\yolo-pipeline.exe pipeline --dataset input\dataset.zip --name helmet-v1 --epochs 100 --imgsz 640
```

The pipeline automatically selects NVIDIA CUDA when PyTorch reports it available; otherwise it uses CPU. Use `--device 0` to require the first CUDA GPU or `--device cpu` to force CPU. MPS is macOS-only and must not be passed on Windows. For NVIDIA setup, install the matching Windows PyTorch/CUDA wheel before running the project setup, following the [official PyTorch installer](https://pytorch.org/get-started/locally/).

Native Windows jobs also use the repository-local `input\`, `workspace\`, and `artifacts\` directories.

### Live screen inference (native macOS and Windows)

Use the trained `.pt` model to detect objects directly from one of your displays:

```sh
.venv/bin/yolo-pipeline predict-screen --model artifacts/jobs/JOB_ID/model/best.pt
```

The display is shown in a preview window with bounding boxes; press `q` or `Esc` to stop.
Keep working in the original application window and press the global `` ` `` key wherever the
pointer is. This records its position and the centre/distance for every detected `head` in
`artifacts/jobs/JOB_ID/screen-clicks.jsonl`. Use `--target-class NAME` to measure another model
class, `--hotkey KEY` to change the trigger, or `--output PATH` for another JSONL.
`--monitor 1` is the first physical display (`--monitor 2` is the next); use `--conf 0.4`,
`--imgsz 640`, `--device mps` (macOS), or `--device 0` (Windows CUDA) as needed. On macOS, enable **Screen Recording** plus **Accessibility/Input Monitoring** for the terminal app (or IDE) in System Settings, then restart it. On Windows, run this command at the same privilege level as the target application; a normal process cannot control an elevated application, the lock screen, or secure UAC desktop. Windows processes enable per-monitor DPI awareness before capture so pointer and frame coordinates remain aligned. This feature must run natively—Docker containers cannot access the host desktop this way.

On Windows, bind recording to the mouse's Next Page side button with `--trigger mouse-forward`.
Use `--window-title "text in the window title"` to capture only one visible window; the title
match must be unique. For example:

```powershell
.venv\Scripts\yolo-pipeline.exe predict-screen artifacts\best.pt --trigger mouse-forward --window-title "My Game"
```

### Docker

CPU (recommended without NVIDIA):

```sh
docker compose -f compose.yaml -f compose.cpu.yaml run --rm yolo-pipeline pipeline --dataset /workspace/input/dataset.zip --name helmet-v1 --epochs 100 --imgsz 640
```

GPU requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```sh
docker compose run --rm yolo-pipeline pipeline --dataset /workspace/input/dataset.zip --name helmet-v1 --video /workspace/input/test.mp4 --model yolo26n.pt
```

Training only: `docker compose run --rm yolo-pipeline train --dataset /workspace/input/dataset.zip --name helmet-v1`.
Video inference: `docker compose run --rm yolo-pipeline predict-video --model /workspace/artifacts/jobs/JOB_ID/model/best.pt --source /workspace/input/test.mp4`.
Inspect a job: `docker compose run --rm yolo-pipeline job show --job-id JOB_ID`.

Artifacts are under `artifacts/jobs/JOB_ID`: models and checksum, reports, optional annotated video plus JSONL, log/manifest. Job working copies remain under `workspace/JOB_ID`; original input is never moved or changed. Resume with `train --resume /workspace/artifacts/jobs/JOB_ID/model/last.pt --dataset ...`.

Load the promoted model with `from ultralytics import YOLO; model = YOLO('best.pt')`. Ultralytics is offered under AGPL-3.0 and Enterprise licenses; review the [Ultralytics license](https://www.ultralytics.com/license) for your use.

## Troubleshooting

Check `dataset-report.txt` first for path/line-specific failures. Ensure the container can write the mounted directories as UID 10001. For GPU failures verify `docker run --gpus all` works and the NVIDIA toolkit is installed. To build/run checks: `make build-cpu`, `make lint`, and `make test`.
