"""Generate a descriptive data-quality / drift report as JSON.

Usage: ``python scripts/monitor.py --help``

The default run scores the explicitly selected dataset, optionally compares it
to a previously written baseline snapshot, and writes both
``artifacts/monitoring_report.json`` and (with ``--write-baseline``)
``artifacts/monitoring_baseline.json``. No external services are required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

# Make ``python scripts/monitor.py`` work when invoked from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraud_ml.config import settings  # noqa: E402
from fraud_ml.data.ingestion import load_dataset  # noqa: E402
from monitoring.drift import create_report, create_snapshot  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "api.yaml"


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


def load_monitoring_config(path: Path | None = None) -> dict[str, Any]:
    """Load monitoring defaults from YAML (missing file means built-in defaults)."""
    return read_yaml_mapping(path or DEFAULT_CONFIG)


def write_json(report: Mapping[str, Any], path: Path) -> None:
    """Atomically-ish write a JSON report, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> dict[str, Any] | None:
    """Read a JSON baseline, returning None when it does not exist."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def optional_integration_notes(config: Mapping[str, Any]) -> list[str]:
    """Report optional dependency status without making any of them required."""
    monitoring_config = config.get("monitoring") or {}
    notes: list[str] = []
    if bool(monitoring_config.get("pandera", False)):
        try:
            import pandera  # noqa: F401

            notes.append("pandera: installed, but no Pandera schema is enforced yet")
        except ImportError:
            notes.append("pandera: NOT INSTALLED - built-in fraud_ml validation used")

    if bool(monitoring_config.get("log_to_mlflow", False)):
        try:
            import mlflow  # noqa: F401

            notes.append(
                "mlflow: installed, but no tracking call is wired in; JSON report is the record"
            )
        except ImportError:
            notes.append("mlflow: NOT INSTALLED - JSON report is the record")

    return notes


def resolve_options(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    """Merge CLI flags, YAML config and environment-backed settings."""
    monitoring_config = config.get("monitoring") or {}
    artifacts_config = config.get("artifacts") or {}
    return {
        "mode": args.mode or settings.data_mode,
        "data_dir": args.data_dir or settings.data_dir,
        "seed": args.seed if args.seed is not None else settings.random_seed,
        "sample_rows": args.sample_rows,
        "baseline_path": args.baseline
        or Path(monitoring_config.get("baseline_path", "artifacts/monitoring_baseline.json")),
        "report_path": args.output
        or Path(monitoring_config.get("report_path", "artifacts/monitoring_report.json")),
        "rate_threshold": args.rate_threshold
        if args.rate_threshold is not None
        else float(monitoring_config.get("rate_absolute_threshold", 0.05)),
        "amount_threshold": args.amount_threshold
        if args.amount_threshold is not None
        else float(monitoring_config.get("amount_relative_threshold", 0.20)),
        "artifact_path": Path(
            artifacts_config.get("model_path", settings.artifact_path)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Define the monitoring CLI surface."""
    parser = argparse.ArgumentParser(
        description="Write a descriptive JSON data-quality/drift report for a dataset batch."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path.")
    parser.add_argument(
        "--mode",
        choices=("demo", "real"),
        default=None,
        help="Explicit dataset mode; overrides config/env (default: demo).",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Dataset directory.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42).")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional row cap.")
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline snapshot JSON path.")
    parser.add_argument("--output", type=Path, default=None, help="Report JSON path.")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write the current snapshot as the baseline if none exists.",
    )
    parser.add_argument(
        "--force-baseline",
        action="store_true",
        help="Overwrite an existing baseline when --write-baseline is set.",
    )
    parser.add_argument(
        "--rate-threshold",
        type=float,
        default=None,
        help="Absolute delta threshold for fraud/missing rates (default: 0.05).",
    )
    parser.add_argument(
        "--amount-threshold",
        type=float,
        default=None,
        help="Relative delta threshold for amount mean/p95 (default: 0.20).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    config = load_monitoring_config(args.config)
    options = resolve_options(args, config)

    frame, source = load_dataset(
        mode=str(options["mode"]),
        data_dir=options["data_dir"],
        seed=int(options["seed"]),
        sample_rows=options["sample_rows"],
    )

    baseline_path: Path = options["baseline_path"]
    baseline = read_json_mapping(baseline_path)
    baseline_written = False
    if args.write_baseline and (baseline is None or args.force_baseline):
        write_json(create_snapshot(frame, str(source)), baseline_path)
        baseline = read_json_mapping(baseline_path)
        baseline_written = True

    report = create_report(
        frame,
        source=str(source),
        baseline=baseline,
        rate_threshold=float(options["rate_threshold"]),
        amount_threshold=float(options["amount_threshold"]),
    )
    report["optional_integrations"] = optional_integration_notes(config)
    report["model_artifact"] = {
        "path": str(options["artifact_path"]),
        "exists": options["artifact_path"].is_file(),
    }
    write_json(report, options["report_path"])

    print(
        json.dumps(
            {
                "report_path": str(options["report_path"]),
                "baseline_path": str(baseline_path),
                "baseline_written": baseline_written,
                "rows": report["rows"],
                "source": report["source"],
                "drift_status": report["drift"].get("status", "compared"),
                "any_exceeded": report["drift"].get("any_exceeded", False),
                "optional_integrations": report["optional_integrations"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
