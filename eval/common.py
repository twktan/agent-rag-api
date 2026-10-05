"""Shared helpers for the evaluation scripts."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "data" / "golden.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


def load_golden(split: str = "test", path: Path = GOLDEN_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    return items if split == "all" else [i for i in items if i["split"] == split]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=EVAL_DIR,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def log_to_mlflow(run_name: str, params: dict, metrics: dict, artifact: Path | None = None) -> None:
    """Log an eval run to MLflow if MLFLOW_TRACKING_URI is configured; otherwise no-op."""
    from app.config import get_settings

    settings = get_settings()
    if not settings.mlflow_tracking_uri:
        return
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"{settings.mlflow_experiment}-eval")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        if artifact is not None:
            mlflow.log_artifact(str(artifact))
