.PHONY: build build-cpu test lint format shell pipeline clean-workspace clean-artifacts
build:
	docker compose build
build-cpu:
	docker compose -f compose.yaml -f compose.cpu.yaml build
test:
	docker compose -f compose.yaml -f compose.cpu.yaml run --rm yolo-pipeline pytest
lint:
	docker compose -f compose.yaml -f compose.cpu.yaml run --rm yolo-pipeline ruff check .
format:
	docker compose -f compose.yaml -f compose.cpu.yaml run --rm yolo-pipeline ruff format .
shell:
	docker compose -f compose.yaml -f compose.cpu.yaml run --rm --entrypoint sh yolo-pipeline
pipeline:
	docker compose run --rm yolo-pipeline pipeline --dataset /workspace/input/dataset.zip --name example
clean-workspace:
	@test -n "$(JOB_ID)" || (echo "Set JOB_ID=<job id>"; exit 2)
	docker compose run --rm --entrypoint sh yolo-pipeline -c 'rm -rf /workspace/jobs/$(JOB_ID)'
clean-artifacts:
	@test -n "$(JOB_ID)" || (echo "Set JOB_ID=<job id>"; exit 2)
	docker compose run --rm --entrypoint sh yolo-pipeline -c 'rm -rf /workspace/artifacts/jobs/$(JOB_ID)'
