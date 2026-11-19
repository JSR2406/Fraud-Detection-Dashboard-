"""Fast, fully offline tests for the FastAPI real-time inference layer."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from api import dependencies, security
from api.dependencies import reset_predictor_cache
from api.main import MAX_REQUEST_BYTES, app
from fraud_ml import config
from fraud_ml.data.ingestion import make_demo_data
from fraud_ml.features.engineering import CATEGORICAL, NUMERIC, engineer_features
from fraud_ml.features.pipeline import build_preprocessor

API_KEY = "demo-test-key"
AUTH = {"X-API-Key": API_KEY}
MISSING_ARTIFACT = Path("missing/candidate.joblib")

PAYLOAD = {
    "transaction_id": "tx-001",
    "TransactionAmt": 42.5,
    "TransactionDT": 86400,
    "ProductCD": "W",
    "card4": "visa",
    "DeviceType": "mobile",
}
RESPONSE_KEYS = {
    "transaction_id",
    "fraud_probability",
    "risk_level",
    "decision",
    "model_version",
}


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    """A tiny fitted pipeline persisted as a trusted bundle."""
    raw = make_demo_data(250, seed=17)
    x = engineer_features(raw.drop(columns="isFraud"))
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(x, raw["isFraud"])
    bundle = {
        "pipeline": pipeline,
        "model_name": "Logistic Regression",
        "threshold": 0.4,
        "feature_columns": list(x.columns),
        "artifact_version": "test-v1",
    }
    path = tmp_path / "candidate.joblib"
    joblib.dump(bundle, path)
    return path


@pytest.fixture(autouse=True)
def _clean_predictor_cache():
    reset_predictor_cache()
    yield
    reset_predictor_cache()


def configure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    artifact_path: Path | None = None,
    api_key: str = API_KEY,
    max_batch_size: int = 2,
    data_mode: str = "demo",
) -> None:
    """Replace the frozen settings everywhere a module holds a reference."""
    fake = replace(
        config.settings,
        artifact_path=MISSING_ARTIFACT if artifact_path is None else artifact_path,
        api_key=api_key,
        max_batch_size=max_batch_size,
        data_mode=data_mode,
    )
    monkeypatch.setattr(config, "settings", fake)
    monkeypatch.setattr(dependencies, "settings", fake)
    monkeypatch.setattr(security, "settings", fake)


def test_live_health_is_always_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/health/live")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["timestamp"]


def test_ready_health_transitions_when_artifact_appears(
    monkeypatch: pytest.MonkeyPatch, artifact: Path
) -> None:
    configure(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert "not available" in response.json()["detail"]

        configure(monkeypatch, artifact_path=artifact)
        reset_predictor_cache()
        response = client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["model_version"] == "test-v1"
        assert body["artifact"] == "candidate.joblib"


def test_authentication_behaviour(monkeypatch: pytest.MonkeyPatch, artifact: Path) -> None:
    configure(monkeypatch, artifact_path=artifact, api_key="")
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=PAYLOAD, headers=AUTH)
        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]

    reset_predictor_cache()
    configure(monkeypatch, artifact_path=artifact)
    with TestClient(app) as client:
        assert client.post("/api/v1/predict", json=PAYLOAD).status_code == 401
        assert (
            client.post(
                "/api/v1/predict", json=PAYLOAD, headers={"X-API-Key": "wrong"}
            ).status_code
            == 401
        )
        assert client.post("/api/v1/predict", json=PAYLOAD, headers=AUTH).status_code == 200


def test_predict_returns_exact_schema(monkeypatch: pytest.MonkeyPatch, artifact: Path) -> None:
    configure(monkeypatch, artifact_path=artifact)
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=PAYLOAD, headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_KEYS
    assert body["transaction_id"] == "tx-001"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["decision"] in {"APPROVE", "REVIEW"}
    assert body["model_version"] == "test-v1"


def test_predict_rejects_extra_fields(monkeypatch: pytest.MonkeyPatch, artifact: Path) -> None:
    configure(monkeypatch, artifact_path=artifact)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict", json={**PAYLOAD, "private_extra": 1}, headers=AUTH
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_batch_respects_max_size(monkeypatch: pytest.MonkeyPatch, artifact: Path) -> None:
    configure(monkeypatch, artifact_path=artifact, max_batch_size=2)
    rows = [{**PAYLOAD, "transaction_id": f"tx-{i}"} for i in range(3)]
    with TestClient(app) as client:
        too_many = client.post("/api/v1/predict/batch", json=rows, headers=AUTH)
        assert too_many.status_code == 422
        assert "maximum" in too_many.json()["detail"]

        response = client.post("/api/v1/predict/batch", json=rows[:2], headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2
        assert all(set(item) == RESPONSE_KEYS for item in body["predictions"])

        envelope = client.post(
            "/api/v1/predict/batch", json={"transactions": rows[:2]}, headers=AUTH
        )
        assert envelope.status_code == 200
        assert envelope.json()["count"] == 2


def test_model_info_returns_version_metadata(
    monkeypatch: pytest.MonkeyPatch, artifact: Path
) -> None:
    configure(monkeypatch, artifact_path=artifact, data_mode="demo")
    with TestClient(app) as client:
        response = client.get("/api/v1/model/info")
        alias = client.get("/model/info")
    assert response.status_code == 200
    assert alias.status_code == 200
    body = response.json()
    assert body["model_name"] == "Logistic Regression"
    assert body["model_version"] == "test-v1"
    assert body["threshold"] == 0.4
    assert body["feature_count"] == len(NUMERIC) + len(CATEGORICAL)
    assert body["artifact"] == "candidate.joblib"
    assert body["data_mode"] == "demo"
    assert str(artifact) not in json.dumps(body)
    assert body == alias.json()


def test_request_id_header_is_always_present(
    monkeypatch: pytest.MonkeyPatch, artifact: Path
) -> None:
    configure(monkeypatch, artifact_path=artifact)
    with TestClient(app) as client:
        generated = client.get("/health/live")
        assert generated.headers.get("X-Request-ID")

        echoed = client.get("/health/live", headers={"X-Request-ID": "trace-123"})
        assert echoed.headers["X-Request-ID"] == "trace-123"

        unauthenticated = client.post("/api/v1/predict", json=PAYLOAD)
        assert unauthenticated.headers.get("X-Request-ID")
        assert unauthenticated.json()["error"]["request_id"]


def test_hard_request_size_cap(monkeypatch: pytest.MonkeyPatch, artifact: Path) -> None:
    configure(monkeypatch, artifact_path=artifact)
    monkeypatch.setattr("api.main.MAX_REQUEST_BYTES", 256)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict", json={**PAYLOAD, "ProductCD": "W" * 500}, headers=AUTH
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    assert response.headers.get("X-Request-ID")


def test_unauthenticated_scoring_never_loads_a_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=PAYLOAD)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"
