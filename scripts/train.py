"""Train fraud candidates and persist the selected artifact.

Usage: ``python scripts/train.py --help``

This CLI delegates model fitting to ``fraud_ml.models.train.train_models`` and
prints a short JSON summary. Artifacts are written to
``<artifact-dir>/candidate.joblib`` and ``<artifact-dir>/metrics.json``.

Optuna, MLflow and Pandera are optional. If they are absent (or disabled in the
YAML config) the run continues with the built-in trainer and validation, and
the skipped integrations are reported in the console summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

# Make ``python scripts/train.py`` work when invoked from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraud_ml.config import settings  # noqa: E402
from fraud_ml.data.ingestion import load_dataset  # noqa: E402
from fraud_ml.models.train import train_models  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "training.yaml"


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, returning an empty mapping when unavailable."""
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        print(f"[warn] PyYAML unavailable; ignoring config file {path}", file=sys.stderr)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return data


def load_training_config(path: Path | None = None) -> dict[str, Any]:
    """Load training defaults from YAML (missing file means library defaults)."""
    return read_yaml_mapping(path or DEFAULT_CONFIG)


def optional_integration_notes(config: Mapping[str, Any], options: Mapping[str, Any]) -> list[str]:
    """Report optional dependency status and how each integration was used."""
    notes: list[str] = []
    tuning_config = config.get("tuning") or {}
    if bool(tuning_config.get("enabled", False)):
        if int(options["tune_trials"]) > 0:
            try:
                import optuna  # noqa: F401

                notes.append("optuna: TPE search used for LightGBM tuning")
            except ImportError:
                notes.append(
                    "optuna: NOT INSTALLED - bounded numpy random search used for LightGBM tuning"
                )
        else:
            notes.append("tuning: disabled (trials = 0); fixed hyperparameters used")
    else:
        notes.append("tuning: disabled in config; fixed hyperparameters used")

    if bool((config.get("experiment_tracking") or {}).get("enabled", False)):
        try:
            import mlflow  # noqa: F401

            notes.append("mlflow: experiment tracking used")
        except ImportError:
            notes.append("mlflow: NOT INSTALLED - local JSON tracking fallback used")

    if bool((config.get("validation") or {}).get("pandera", False)) or options.get(
        "use_pandera", False
    ):
        installed, detail = pandera_available()
        if options.get("use_pandera", False) and installed:
            notes.append(f"pandera: schema enforced on the loaded dataset (v{detail})")
        elif not installed:
            notes.append(f"pandera: NOT INSTALLED - built-in validation used ({detail})")
        else:
            notes.append("pandera: disabled; built-in fraud_ml validation used")

    notes.append(
        f"calibration: {options['calibration_method']} fitted on validation; "
        f"threshold strategy: {options['threshold_strategy']}"
    )
    return notes


def resolve_options(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge CLI flags, YAML config and environment-backed settings."""
    data_config = config.get("data") or {}
    artifact_config = config.get("artifacts") or {}
    threshold_config = config.get("threshold") or {}
    calibration_config = config.get("calibration") or {}
    tuning_config = config.get("tuning") or {}
    registry_config = config.get("registry") or {}
    tracking_config = config.get("experiment_tracking") or {}
    default_artifact_dir = Path(settings.artifact_path).parent

    tune_enabled = args.tune if args.tune is not None else bool(tuning_config.get("enabled", False))
    trials = (
        args.trials
        if args.trials is not None
        else int(tuning_config.get("trials", 12))
    )

    return {
        "mode": args.mode or str(data_config.get("mode", settings.data_mode)),
        "data_dir": args.data_dir or Path(data_config.get("data_dir", settings.data_dir)),
        "seed": args.seed if args.seed is not None else int(data_config.get("seed", settings.random_seed)),
        "sample_rows": args.sample_rows if args.sample_rows is not None else data_config.get("sample_rows"),
        "artifact_dir": args.artifact_dir
        or Path(artifact_config.get("dir", default_artifact_dir)),
        "threshold_config": dict(threshold_config),
        "threshold_strategy": str(threshold_config.get("strategy", settings.threshold_strategy)),
        "calibration_method": str(calibration_config.get("method", settings.calibration_method)),
        "tune_trials": int(trials) if tune_enabled else 0,
        "tuning_study_name": str(tuning_config.get("study_name", "fraud-lightgbm")),
        "tuning_study_storage": tuning_config.get("storage"),
        "mlflow_tracking_uri": args.tracking_uri or tracking_config.get("tracking_uri") or None,
        "register": bool(registry_config.get("register", True)),
    }


def run_training(options: Mapping[str, Any]) -> dict[str, Any]:
    """Load the explicitly selected dataset, train, and read back the report."""
    from fraud_ml.models.business import BusinessAssumptions

    frame, source = load_dataset(
        mode=str(options["mode"]),
        data_dir=options["data_dir"],
        seed=int(options["seed"]),
        sample_rows=options["sample_rows"],
    )
    threshold_config = options.get("threshold_config", {})
    assumptions = BusinessAssumptions(
        false_positive_cost=float(threshold_config.get("false_positive_cost", 50.0)),
        missed_fraud_cost=float(threshold_config.get("missed_fraud_cost", 500.0)),
        amount_at_risk_weight=float(threshold_config.get("amount_at_risk_weight", 1.0)),
        investigation_capacity=threshold_config.get("investigation_capacity"),
    )
    result = train_models(
        frame,
        seed=int(options["seed"]),
        artifact_dir=options["artifact_dir"],
        threshold_strategy=options["threshold_strategy"],
        assumptions=assumptions,
        tune_lightgbm_trials=int(options["tune_trials"]),
        calibration_method=options["calibration_method"],
        register=bool(options["register"]),
        tuning_study_name=str(options["tuning_study_name"]),
        tuning_study_storage=options["tuning_study_storage"],
        mlflow_tracking_uri=options["mlflow_tracking_uri"],
    )
    metrics_path = Path(options["artifact_dir"]) / "metrics.json"
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "source": str(source),
        "mode": str(options["mode"]),
        "rows": int(len(frame)),
        "best_name": result["best_name"],
        "artifact_path": result["artifact_path"],
        "metrics_path": str(metrics_path),
        "tracking_backend": result["tracking"]["backend"],
        "mlflow_run_id": result["tracking"]["run_id"],
        "tuning": result["tuning"],
        "gates": result["gates"],
        "report": report,
    }


def build_parser() -> argparse.ArgumentParser:
    """Define the training CLI surface."""
    parser = argparse.ArgumentParser(
        description=(
            "Train fraud candidates on an explicit real/demo dataset and write "
            "artifacts/candidate.joblib plus artifacts/metrics.json."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--mode",
        choices=("demo", "real"),
        default=None,
        help="Explicit dataset mode; overrides config/env (default: demo).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset directory; required for --mode real (default: data).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42).")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row cap for smoke runs; null in config means all rows.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory for candidate.joblib and metrics.json (default: artifacts).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=None,
        help="Enable Optuna LightGBM tuning (overrides configs/training.yaml).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of Optuna trials when tuning is enabled (default: config value).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI, e.g. sqlite:///mlflow.db (overrides config/env).",
    )
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Also print the full metrics.json report to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    config = load_training_config(args.config)
    options = resolve_options(args, config)
    summary = run_training(options)

    console = {key: value for key, value in summary.items() if key != "report"}
    console["optional_integrations"] = optional_integration_notes(config, options)
    print(json.dumps(console, indent=2))
    if args.print_metrics:
        print(json.dumps(summary["report"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
