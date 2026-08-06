ARG ULTRALYTICS_IMAGE=ultralytics/ultralytics:8.4.115
FROM ${ULTRALYTICS_IMAGE}
WORKDIR /app
COPY requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.lock
COPY app ./app
COPY configs ./configs
COPY docker ./docker
RUN useradd --create-home --uid 10001 appuser && chmod +x docker/entrypoint.sh && mkdir -p /workspace/input /workspace/jobs /workspace/artifacts && chown -R appuser:appuser /app /workspace
USER appuser
ENV PYTHONUNBUFFERED=1 \
    YOLO_INPUT_DIR=/workspace/input \
    YOLO_JOBS_DIR=/workspace/jobs \
    YOLO_ARTIFACTS_DIR=/workspace/artifacts
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["yolo-pipeline", "--help"]
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python /app/docker/healthcheck.py
