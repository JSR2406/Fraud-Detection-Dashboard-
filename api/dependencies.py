"""Shared FastAPI dependencies for the real-time inference layer.

A trusted joblib artifact is deserialized at most once per process.  Any load
failure is reported as a 503 so liveness stays green while readiness and the
scoring endpoints clearly state that the model is unavailable.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from fraud_ml.config import Settings, settings as _default_settings
from fraud_ml.inference.predictor import FraudPredictor

# Operators/tests replace this frozen Settings object rather than mutating it.
settings: Settings = _default_settings

# Module-level cache: the model is loaded exactly once per process.
_predictor_cache: FraudPredictor | None = None


def get_settings() -> Settings:
    """Return the active settings, honouring module-level replacements."""
    current = globals()["settings"]
    if current is _default_settings:
        # No explicit override here: respect a replacement made in
        # fraud_ml.config (Settings is frozen and must be replaced wholesale).
        from fraud_ml import config

        return config.settings
    return current


def reset_predictor_cache() -> None:
    """Drop the cached predictor (for tests and trusted artifact rotation)."""
    global _predictor_cache
    _predictor_cache = None


def get_predictor() -> FraudPredictor:
    """Lazily return the single trusted predictor or raise a clear 503."""
    global _predictor_cache
    if _predictor_cache is None:
        artifact = Path(get_settings().artifact_path)
        try:
            _predictor_cache = FraudPredictor.from_trusted_file(artifact)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model artifact '{artifact.name}' is not available. "
                    "Provision a trusted artifact before scoring."
                ),
            ) from exc
        except Exception as exc:
            # joblib deserialization can raise many exception types; none of
            # their messages or absolute paths are safe to expose to clients.
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model artifact '{artifact.name}' could not be loaded; "
                    "it is missing or invalid."
                ),
            ) from exc
    return _predictor_cache


def load_predictor() -> bool:
    """Eagerly load at startup; never raise because an artifact is absent."""
    try:
        get_predictor()
    except HTTPException:
        return False
    return True
