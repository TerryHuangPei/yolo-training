# YOLO Training Pipeline

Container-first automation for validating YOLO detection datasets, training Ultralytics YOLO, promoting `best.pt`, and streaming video inference. It requires Python 3.11 inside the supplied Docker image; no host Python is used.

## Input datasets

Split datasets contain `images/train`, `images/val`, parallel `labels/train`, `labels/val`, and `dataset.yaml`. Unsplit datasets contain `images`, `labels`, and `classes.txt`; they are copied to the job workspace and split 80/20 with seed 42. Labels use `class_id x_center y_center width height`, normalized to 0–1. Empty images may omit a label.

Validation detects corrupt images, malformed labels, invalid classes/boxes, missing counterparts, exact duplicate images, split leakage, Zip Slip, ZIP symlinks, limits, and compression bombs. Reports are written to each job.

## Run

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
