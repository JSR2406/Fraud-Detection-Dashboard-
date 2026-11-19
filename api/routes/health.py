"""Liveness and readiness probes for the fraud inference API."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from typing_extensions import TypedDict

from api.dependencies import get_predictor, get_settings
from fraud_ml.config import Settings

router = APIRouter(tags=["health"])


class LiveResponse(TypedDict):
    status: str
    timestamp: str


class ReadyResponse(TypedDict):
    status: str
    timestamp: str
    model_version: str
    artifact: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.get("/health/live", response_model=LiveResponse)
def live() -> LiveResponse:
    """Always-OK process liveness probe; deliberately independent of the model."""
    return {"status": "ok", "timestamp": _now()}


@router.get("/health/ready", response_model=ReadyResponse)
def ready(settings: Settings = Depends(get_settings)) -> ReadyResponse:
    """Report readiness only when a trusted artifact exists and can be loaded."""
    artifact = Path(settings.artifact_path)
    if not artifact.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact '{artifact.name}' is not available; service is not ready.",
        )
    predictor = get_predictor()
    return {
        "status": "ready",
        "timestamp": _now(),
        "model_version": predictor.model_version,
        "artifact": artifact.name,
    }
