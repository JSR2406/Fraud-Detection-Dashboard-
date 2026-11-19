"""Experiment tracking with MLflow and a local JSON fallback.

MLflow is optional: when unavailable, runs are still recorded locally with a
dataset fingerprint, code revision, metrics and config so experiments remain
auditable. The fallback is never described as an MLflow run.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TrackingResult:
    backend: str
    run_id: str
    tracking_uri: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "run_id": self.run_id,
            "tracking_uri": self.tracking_uri,
            "details": self.details,
        }


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of schema, row count and sampled content (not a version tag)."""
    buffer = "\n".join([str(list(df.columns)), str(len(df)), str(df.dtypes.tolist())])
    digest = hashlib.sha256(buffer.encode("utf-8")).hexdigest()[:16]
    sample = df.head(100).to_csv(index=False).encode("utf-8")
    return f"{digest}-{hashlib.sha256(sample).hexdigest()[:8]}"


def code_revision() -> str:
    """Best-effort git revision; never raises in non-git deployments."""
    import subprocess

    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, timeout=3
        )
        return output.decode("utf-8", errors="replace").strip()
    except Exception:
        return os.getenv("FRAUD_CODE_REVISION", "unknown")


def log_training_run(
    *,
    experiment: str,
    run_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    artifact_paths: list[str | Path] | None = None,
    output_dir: str | Path = "artifacts/ml_tracking",
) -> TrackingResult:
    """Log a training run to MLflow when installed, else to local JSON files."""
    tags = dict(tags or {})
    tags.setdefault("dataset_fingerprint", tags.get("dataset_fingerprint", "unknown"))
    tags.setdefault("code_revision", code_revision())

    try:
        import mlflow  # type: ignore

        tracking_uri = os.getenv("FRAUD_MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as active:
            for key, value in parameters.items():
                mlflow.log_param(key, value)
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(key, float(value))
            mlflow.set_tags(tags)
            for artifact in artifact_paths or []:
                path = Path(artifact)
                if path.exists():
                    mlflow.log_artifact(str(path))
        return TrackingResult(
            backend="mlflow",
            run_id=str(active.info.run_id),
            tracking_uri=tracking_uri,
            details={"experiment": experiment, "tags": tags},
        )
    except ImportError:
        pass

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_id = hashlib.sha256(
        f"{run_name}{json.dumps(parameters, sort_keys=True, default=str)}".encode("utf-8")
    ).hexdigest()[:12]
    payload = {
        "backend": "local-json-fallback (mlflow not installed)",
        "run_id": run_id,
        "experiment": experiment,
        "run_name": run_name,
        "parameters": parameters,
        "metrics": metrics,
        "tags": tags,
        "artifact_paths": [str(Path(path).name) for path in artifact_paths or []],
    }
    target = output / f"{run_id}.json"
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return TrackingResult(
        backend="local-json-fallback",
        run_id=run_id,
        tracking_uri=str(output.resolve()),
        details={"experiment": experiment, "path": str(target)},
    )
