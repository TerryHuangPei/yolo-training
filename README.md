# YOLO Training Pipeline

Automation for validating YOLO detection datasets, training Ultralytics YOLO, promoting `best.pt`, and streaming video inference. It supports Docker on Linux/NVIDIA and native Apple Silicon macOS execution.

## Input datasets

Split datasets contain `images/train`, `images/val`, parallel `labels/train`, `labels/val`, and `dataset.yaml`. Unsplit datasets contain `images`, `labels`, and `classes.txt`; they are copied to the job workspace and split 80/20 with seed 42. Labels use `class_id x_center y_center width height`, normalized to 0–1. Empty images may omit a label.

Validation detects corrupt images, malformed labels, invalid classes/boxes, missing counterparts, exact duplicate images, split leakage, Zip Slip, ZIP symlinks, limits, and compression bombs. Reports are written to each job.

## Run

### Native Apple Silicon macOS

The `mac-version` branch runs directly on a Mac and deliberately does not use the Linux `amd64` Ultralytics image. This avoids Docker's `linux/amd64` versus `linux/arm64/v8` platform mismatch. Python 3.11 is required; install it with Homebrew if needed:

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

### Live screen inference (native macOS)

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
`--imgsz 640`, or `--device mps` as needed. On first use macOS prompts for **Screen Recording** permission. Enable it for the terminal app (or the IDE) that launched the command in **System Settings → Privacy & Security → Screen & System Audio Recording**, then restart the command. The global hotkey also requires **Accessibility** permission for that same app. This feature must run natively on macOS—Docker containers cannot access the host desktop this way.

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
