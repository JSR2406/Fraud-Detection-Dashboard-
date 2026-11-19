"""Start (or preflight-check) the FastAPI scoring service.

Usage: ``python scripts/serve.py --help``

The default import target is ``api.main:app`` on port 8000, matching the
Docker image. The script auto-detects whether the target is an ASGI app or an
app factory and forwards that choice to uvicorn; override it with
``--factory`` / ``--no-factory`` if detection is ambiguous.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

# Make ``python scripts/serve.py`` work when invoked from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraud_ml.config import settings  # noqa: E402

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


def load_api_config(path: Path | None = None) -> dict[str, Any]:
    """Load serving defaults from YAML (missing file means built-in defaults)."""
    return read_yaml_mapping(path or DEFAULT_CONFIG)


def artifact_status(path: Path) -> dict[str, Any]:
    """Describe the trusted model artifact without deserialising it."""
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
    }


def detect_factory(app_spec: str) -> bool:
    """Return True when the import target looks like an app factory.

    Returns False on any import error so uvicorn can produce the authoritative
    error message instead of this preflight helper.
    """
    module_name, separator, attribute = app_spec.partition(":")
    if not separator or not attribute:
        return False
    try:
        target = getattr(importlib.import_module(module_name), attribute)
    except Exception:
        return False
    return callable(target) and not hasattr(target, "routes")


def resolve_server_options(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge CLI flags, environment variables and YAML config."""
    server = config.get("server") or {}
    app_config = config.get("app") or {}
    return {
        "app": args.app or str(app_config.get("target", "api.main:app")),
        "host": args.host or os.getenv("FRAUD_API_HOST", str(server.get("host", "0.0.0.0"))),
        "port": args.port
        if args.port is not None
        else int(os.getenv("FRAUD_API_PORT", str(server.get("port", 8000)))),
        "log_level": args.log_level or str(server.get("log_level", "info")),
        "workers": args.workers
        if args.workers is not None
        else int(server.get("workers", 1)),
        "reload": args.reload
        if args.reload is not None
        else bool(server.get("reload", False)),
    }


def build_parser() -> argparse.ArgumentParser:
    """Define the serving CLI surface."""
    parser = argparse.ArgumentParser(
        description="Run the FastAPI fraud scoring service with uvicorn."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path.")
    parser.add_argument("--app", default=None, help="Uvicorn import target (default: api.main:app).")
    parser.add_argument("--host", default=None, help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: 8000).")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Trusted model artifact to report (default: FRAUD_MODEL_PATH).",
    )
    parser.add_argument("--log-level", default=None, help="Uvicorn log level (default: info).")
    parser.add_argument("--workers", type=int, default=None, help="Worker count (default: 1).")
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable auto-reload (development only; default from config).",
    )
    parser.add_argument(
        "--factory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Treat --app as an app factory (auto-detected by default).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and artifact presence, then exit without serving.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    config = load_api_config(args.config)
    options = resolve_server_options(args, config)
    artifact = args.artifact or Path(settings.artifact_path)
    model_status = artifact_status(artifact)

    if args.check:
        settings.validate()
        report = {
            "status": "ok" if model_status["exists"] else "artifact_missing",
            "server": options,
            "model": model_status,
            "api_key_configured": bool(settings.api_key),
        }
        print(json.dumps(report, indent=2))
        return 0 if model_status["exists"] else 1

    try:
        import uvicorn
    except ImportError:
        print("[error] uvicorn is not installed; run pip install -r requirements.txt", file=sys.stderr)
        return 2

    if not model_status["exists"]:
        print(
            f"[warn] Model artifact not found at {model_status['path']}; "
            "the API may fail until training output is mounted.",
            file=sys.stderr,
        )

    app_spec = options.pop("app")
    factory = args.factory
    if factory is None:
        factory = detect_factory(app_spec)

    uvicorn.run(app_spec, factory=factory, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
