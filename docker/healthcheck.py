from pathlib import Path

required = (Path("/workspace/input"), Path("/workspace/jobs"), Path("/workspace/artifacts"))
raise SystemExit(0 if all(path.exists() for path in required) else 1)
