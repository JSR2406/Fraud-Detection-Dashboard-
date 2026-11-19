"""Real-time scoring, batch scoring and model metadata endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.dependencies import get_predictor, get_settings
from api.security import require_api_key
from fraud_ml.config import Settings
from fraud_ml.features.engineering import CATEGORICAL, NUMERIC
from fraud_ml.inference.predictor import FraudPredictor

router = APIRouter(tags=["inference"])

# Root-level alias for metadata, in addition to the prefixed /api/v1 route.
root_info_router = APIRouter(tags=["model"])


class PredictRequest(BaseModel):
    """One transaction. Unknown fields are rejected rather than silently dropped."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    TransactionAmt: float = Field(ge=0, allow_inf_nan=False)
    TransactionDT: float | None = Field(default=None, allow_inf_nan=False)
    ProductCD: str | None = None
    card1: float | None = Field(default=None, allow_inf_nan=False)
    card2: float | None = Field(default=None, allow_inf_nan=False)
    card4: str | None = None
    card6: str | None = None
    P_emaildomain: str | None = None
    dist1: float | None = Field(default=None, allow_inf_nan=False)
    C1: float | None = Field(default=None, allow_inf_nan=False)
    C2: float | None = Field(default=None, allow_inf_nan=False)
    C6: float | None = Field(default=None, allow_inf_nan=False)
    C11: float | None = Field(default=None, allow_inf_nan=False)
    V95: float | None = Field(default=None, allow_inf_nan=False)
    V96: float | None = Field(default=None, allow_inf_nan=False)
    V97: float | None = Field(default=None, allow_inf_nan=False)
    DeviceType: str | None = None

    def as_row(self) -> dict:
        """Omit absent fields so the feature builder can supply its defaults."""
        return self.model_dump(exclude_none=True)


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW"]
    model_version: str


class BatchEnvelope(BaseModel):
    """Optional object form of the batch body: {"transactions": [...]}."""

    model_config = ConfigDict(extra="forbid")

    transactions: list[PredictRequest]


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    feature_count: int
    artifact: str
    data_mode: str


def _build_model_info(predictor: FraudPredictor, settings: Settings) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        threshold=predictor.threshold,
        feature_count=(
            len(predictor.feature_columns)
            if predictor.feature_columns is not None
            else len(NUMERIC) + len(CATEGORICAL)
        ),
        artifact=Path(settings.artifact_path).name,
        data_mode=settings.data_mode,
    )


@router.get("/model/info", response_model=ModelInfoResponse, operation_id="model_info_v1")
def model_info(
    predictor: FraudPredictor = Depends(get_predictor),
    settings: Settings = Depends(get_settings),
) -> ModelInfoResponse:
    """Expose metadata only — never full local paths or other secrets."""
    return _build_model_info(predictor, settings)


@root_info_router.get(
    "/model/info", response_model=ModelInfoResponse, operation_id="model_info_root"
)
def model_info_root(
    predictor: FraudPredictor = Depends(get_predictor),
    settings: Settings = Depends(get_settings),
) -> ModelInfoResponse:
    """Unprefixed alias for the same metadata response."""
    return _build_model_info(predictor, settings)


@router.post("/predict", response_model=PredictionResponse)
def predict_one(
    payload: PredictRequest,
    _auth: None = Depends(require_api_key),
    predictor: FraudPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Score a single transaction with the trusted model."""
    return predictor.predict([payload.as_row()])[0]


@router.post("/predict/batch", response_model=BatchResponse)
def predict_batch(
    payload: list[PredictRequest] | BatchEnvelope,
    _auth: None = Depends(require_api_key),
    predictor: FraudPredictor = Depends(get_predictor),
    settings: Settings = Depends(get_settings),
) -> BatchResponse:
    """Score a bounded list of transactions in one call."""
    rows = payload if isinstance(payload, list) else payload.transactions
    if not rows:
        raise HTTPException(status_code=422, detail="Batch must contain at least one transaction.")
    if len(rows) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch of {len(rows)} exceeds the maximum of {settings.max_batch_size}.",
        )
    predictions = predictor.predict([row.as_row() for row in rows])
    return BatchResponse(count=len(predictions), predictions=predictions)
