"""Experiment tracking with MLflow and a local JSON fallback.

MLflow is optional at import time for the rest of the platform, but when it is
installed this module records parameters, metrics, tags, artifacts and the
fitted sklearn pipeline so runs are reproducible and registrable. Any MLflow
store failure degrades to local JSON with an explicit reason instead of
breaking training.

MLflow 3.x requires a database-backed tracking store (for example
``sqlite:///mlflow.db``); the legacy file store is intentionally not used.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_ml.config import settings


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


def _scalar(value: Any) -> Any:
    """MLflow params/tags accept scalars; JSON-encode nested structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, default=str, sort_keys=True)


def _local_fallback(
    *,
    experiment: str,
    run_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str],
    artifact_paths: list[str | Path],
    output_dir: str | Path,
    reason: str,
) -> TrackingResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_id = hashlib.sha256(
        f"{run_name}{json.dumps(parameters, sort_keys=True, default=str)}".encode("utf-8")
    ).hexdigest()[:12]
    payload = {
        "backend": "local-json-fallback (mlflow unavailable or failed)",
        "run_id": run_id,
        "experiment": experiment,
        "run_name": run_name,
        "parameters": parameters,
        "metrics": metrics,
        "tags": tags,
        "artifact_paths": [str(Path(path).name) for path in artifact_paths],
        "fallback_reason": reason,
    }
    target = output / f"{run_id}.json"
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return TrackingResult(
        backend="local-json-fallback",
        run_id=run_id,
        tracking_uri=str(output.resolve()),
        details={"experiment": experiment, "path": str(target), "reason": reason},
    )


def log_training_run(
    *,
    experiment: str,
    run_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    artifact_paths: list[str | Path] | None = None,
    output_dir: str | Path = "artifacts/ml_tracking",
    model: Any | None = None,
    tracking_uri: str | None = None,
) -> TrackingResult:
    """Log a training run to MLflow when installed, else to local JSON files."""
    tags = {key: str(value) for key, value in (tags or {}).items()}
    tags.setdefault("dataset_fingerprint", "unknown")
    tags.setdefault("code_revision", code_revision())
    artifacts = list(artifact_paths or [])
    uri = tracking_uri or settings.mlflow_tracking_uri

    try:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow  # type: ignore
        import mlflow.sklearn  # type: ignore

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as active:
            mlflow.log_params({key: _scalar(value) for key, value in parameters.items()})
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            )
            mlflow.set_tags(tags)
            for artifact in artifacts:
                path = Path(artifact)
                if path.exists():
                    mlflow.log_artifact(str(path))
            if model is not None:
                # MLflow 3 defaults to skops, which rejects numpy dtypes used by
                # sklearn ColumnTransformers. The model is our own trained object
                # (never untrusted input), so cloudpickle is the compatible format.
                try:
                    mlflow.sklearn.log_model(
                        sk_model=model, name="model", serialization_format="cloudpickle"
                    )
                except TypeError:  # MLflow < 3 compatibility
                    mlflow.sklearn.log_model(
                        sk_model=model, artifact_path="model", serialization_format="cloudpickle"
                    )

        logged_model_id = None
        try:  # MLflow 3 tracks logged models separately from run artifacts
            logged_model_id = getattr(mlflow.last_logged_model(), "model_id", None)
        except Exception:
            pass

        return TrackingResult(
            backend="mlflow",
            run_id=str(active.info.run_id),
            tracking_uri=uri,
            details={
                "experiment": experiment,
                "tags": tags,
                "model_logged": model is not None,
                "logged_model_id": logged_model_id,
            },
        )
    except ImportError:
        reason = "mlflow is not installed"
    except Exception as exc:  # store/network configuration errors must not break training
        reason = f"{type(exc).__name__}: {exc}"

    return _local_fallback(
        experiment=experiment,
        run_name=run_name,
        parameters=parameters,
        metrics=metrics,
        tags=tags,
        artifact_paths=artifacts,
        output_dir=output_dir,
        reason=reason,
    )
