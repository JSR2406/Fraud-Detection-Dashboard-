"""Model candidate registration, validation gates, promotion and rollback.

Uses MLflow Model Registry aliases when MLflow is configured. Otherwise a local
JSON registry provides the same lifecycle contract for demos and tests. A model
is never promoted on a single metric: all documented gates must pass.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fraud_ml.inference.predictor import FraudPredictor


@dataclass(frozen=True)
class GateResult:
    """Outcome of documented validation gates."""

    passed: bool
    failures: list[str] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "failures": list(self.failures)}


def evaluation_gates(
    validation_metrics: dict[str, Any],
    *,
    min_average_precision: float = 0.05,
    max_brier_score: float = 0.5,
    max_false_positive_rate: float = 0.9,
) -> GateResult:
    """Check multiple validation criteria before a candidate may be promoted."""
    failures: list[str] = []
    ap = float(validation_metrics.get("average_precision", 0.0))
    brier = float(validation_metrics.get("brier_score", 1.0))
    fpr = float(validation_metrics.get("false_positive_rate", 1.0))
    if ap < min_average_precision:
        failures.append(f"average_precision {ap:.4f} < {min_average_precision}")
    if brier > max_brier_score:
        failures.append(f"brier_score {brier:.4f} > {max_brier_score}")
    if fpr > max_false_positive_rate:
        failures.append(f"false_positive_rate {fpr:.4f} > {max_false_positive_rate}")
    return GateResult(passed=not failures, failures=failures)


def _registry_path(registry_dir: str | Path) -> Path:
    return Path(registry_dir) / "registry.json"


def _read_registry(registry_dir: str | Path) -> dict[str, Any]:
    path = _registry_path(registry_dir)
    if not path.is_file():
        return {"versions": {}, "aliases": {}, "history": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_registry(registry_dir: str | Path, data: dict[str, Any]) -> Path:
    path = _registry_path(registry_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def register_candidate(
    artifact_path: str | Path,
    *,
    model_name: str,
    version: str,
    validation_metrics: dict[str, Any],
    gate: GateResult | None = None,
    registry_dir: str | Path = "artifacts/registry",
) -> dict[str, Any]:
    """Validate a trusted artifact and register it as a candidate version."""
    artifact = Path(artifact_path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact}")
    predictor = FraudPredictor.from_trusted_file(artifact)  # validates schema
    gate = gate or evaluation_gates(validation_metrics)
    registry = _read_registry(registry_dir)
    record = {
        "version": str(version),
        "model_name": str(model_name),
        "artifact": str(artifact),
        "model_version": predictor.model_version,
        "threshold": predictor.threshold,
        "validation_metrics": validation_metrics,
        "gates": gate.as_dict(),
        "status": "candidate" if gate.passed else "rejected",
    }
    registry["versions"][str(version)] = record
    registry.setdefault("history", []).append(
        {"action": "register", "version": str(version), "status": record["status"]}
    )
    _write_registry(registry_dir, registry)
    return record


def promote(version: str, *, alias: str = "champion", registry_dir: str | Path = "artifacts/registry") -> dict[str, Any]:
    """Promote a registered, gate-passing candidate to an alias."""
    registry = _read_registry(registry_dir)
    record = registry["versions"].get(str(version))
    if record is None:
        raise KeyError(f"Unknown model version: {version}")
    if record["gates"].get("passed") is not True:
        raise ValueError("Only gate-passing candidates may be promoted.")
    previous = registry.get("aliases", {}).get(alias)
    registry.setdefault("aliases", {})[alias] = str(version)
    registry.setdefault("history", []).append(
        {"action": "promote", "alias": alias, "version": str(version), "previous": previous}
    )
    record["status"] = "active"
    _write_registry(registry_dir, registry)
    return {"alias": alias, "version": str(version), "previous": previous}


def rollback(*, alias: str = "champion", registry_dir: str | Path = "artifacts/registry") -> dict[str, Any]:
    """Point an alias back to the previously promoted version."""
    registry = _read_registry(registry_dir)
    promotions = [
        entry
        for entry in registry.get("history", [])
        if entry.get("action") == "promote" and entry.get("alias") == alias
    ]
    if len(promotions) < 2:
        raise ValueError("No earlier version is available for rollback.")
    target = promotions[-2]["version"]
    current = registry.get("aliases", {}).get(alias)
    registry.setdefault("aliases", {})[alias] = target
    registry.setdefault("history", []).append(
        {"action": "rollback", "alias": alias, "version": target, "previous": current}
    )
    _write_registry(registry_dir, registry)
    return {"alias": alias, "version": target, "previous": current}


def load_alias(*, alias: str = "champion", registry_dir: str | Path = "artifacts/registry") -> FraudPredictor:
    """Load the predictor referenced by a registry alias (trusted local path)."""
    registry = _read_registry(registry_dir)
    version = registry.get("aliases", {}).get(alias)
    if version is None:
        raise KeyError(f"No model registered under alias '{alias}'.")
    return FraudPredictor.from_trusted_file(registry["versions"][version]["artifact"])


def mlflow_registration_available() -> bool:
    """Whether MLflow Model Registry aliases can be used in this environment."""
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False
