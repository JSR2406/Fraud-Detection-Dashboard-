"""Real-backend tests for Optuna tuning and MLflow tracking/registry.

These tests exercise the installed MLflow/Optuna backends against temporary
SQLite stores. They are skipped when the packages are not available so CI still
works in minimal environments.
"""
from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_ml.config import settings
from fraud_ml.data.ingestion import make_demo_data
from fraud_ml.data.splitting import chronological_split
from fraud_ml.features.engineering import engineer_features
from fraud_ml.features.pipeline import build_preprocessor
from fraud_ml.mlops import registry as registry_module
from fraud_ml.mlops import tracking as tracking_module
from fraud_ml.mlops.registry import evaluation_gates, promote, register_candidate
from fraud_ml.mlops.tracking import log_training_run
from fraud_ml.models.tuning import tune_lightgbm


def _features_and_labels(rows: int = 700, seed: int = 3):
    parts = chronological_split(make_demo_data(rows, seed=seed))
    x = engineer_features(parts.train.drop(columns="isFraud", errors="ignore"))
    xv = engineer_features(parts.validation.drop(columns="isFraud", errors="ignore"))
    return x, parts.train.isFraud.astype(int), xv, parts.validation.isFraud.astype(int)


def test_optuna_study_is_used_and_persisted(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    import optuna

    x, y, xv, yv = _features_and_labels()
    study_name = f"test-study-{uuid.uuid4().hex[:8]}"
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    result = tune_lightgbm(
        x, y, xv, yv, n_trials=2, seed=11, study_name=study_name, study_storage=storage
    )

    assert result.backend == "optuna", result.backend
    assert len(result.trials) == 2
    assert 0.0 <= result.best_score <= 1.0
    study = optuna.load_study(study_name=study_name, storage=storage)
    assert len(study.trials) == 2
    assert set(result.best_params).issubset(set(study.best_params))


def test_mlflow_tracking_logs_run_params_metrics_and_model(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(tracking_module, "settings", replace(settings, mlflow_tracking_uri=uri))

    x, y, _, _ = _features_and_labels(400)
    pipeline = Pipeline(
        [("preprocessor", build_preprocessor()), ("model", LogisticRegression(max_iter=200))]
    )
    pipeline.fit(x, y)

    result = log_training_run(
        experiment=f"test-exp-{uuid.uuid4().hex[:8]}",
        run_name="tracking-test",
        parameters={"seed": 42, "nested": {"a": 1}},
        metrics={"validation_average_precision": 0.42},
        tags={"purpose": "unit-test"},
        model=pipeline,
    )

    assert result.backend == "mlflow", result.details
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(result.run_id)
    assert run.data.params["seed"] == "42"
    assert run.data.metrics["validation_average_precision"] == pytest.approx(0.42)
    assert run.data.tags["purpose"] == "unit-test"
    # MLflow 3 records logged models separately from run artifacts.
    logged_models = client.search_logged_models(experiment_ids=[run.info.experiment_id])
    assert [model.model_id for model in logged_models], logged_models


def test_mlflow_registry_registers_and_sets_alias(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("mlflow")
    import joblib
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(registry_module, "settings", replace(settings, mlflow_tracking_uri=uri))
    monkeypatch.setattr(tracking_module, "settings", replace(settings, mlflow_tracking_uri=uri))

    x, y, _, _ = _features_and_labels(400)
    pipeline = Pipeline(
        [("preprocessor", build_preprocessor()), ("model", LogisticRegression(max_iter=200))]
    )
    pipeline.fit(x, y)
    artifact = tmp_path / "candidate.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "model_name": "test",
            "threshold": 0.4,
            "feature_columns": list(x.columns),
            "artifact_version": "test-v1",
        },
        artifact,
    )

    run = log_training_run(
        experiment=f"test-exp-{uuid.uuid4().hex[:8]}",
        run_name="registry-test",
        parameters={"seed": 42},
        metrics={"validation_average_precision": 0.6},
        model=pipeline,
    )
    model_name = f"fraud-test-{uuid.uuid4().hex[:8]}"
    gate = evaluation_gates({"average_precision": 0.6, "brier_score": 0.1, "false_positive_rate": 0.2})
    record = register_candidate(
        artifact,
        model_name="test",
        version="1",
        validation_metrics={"average_precision": 0.6},
        gate=gate,
        registry_dir=tmp_path / "registry",
        mlflow_run_id=run.run_id,
        mlflow_model_name=model_name,
    )

    assert record.get("mlflow_version") is not None, record
    client = MlflowClient(tracking_uri=uri)
    version = client.get_model_version(model_name, record["mlflow_version"])
    assert version.tags["gate_passed"] == "True"

    promoted = promote("1", registry_dir=tmp_path / "registry")
    assert promoted["mlflow_version"] == record["mlflow_version"]
    aliased = client.get_model_version_by_alias(model_name, "champion")
    assert str(aliased.version) == str(record["mlflow_version"])


def test_train_models_with_optuna_and_mlflow(tmp_path: Path, monkeypatch) -> None:
    """Integration check: tuning feeds the model and the run lands in MLflow."""
    pytest.importorskip("optuna")
    pytest.importorskip("mlflow")
    from fraud_ml.models.train import train_models

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(tracking_module, "settings", replace(settings, mlflow_tracking_uri=uri))
    monkeypatch.setattr(registry_module, "settings", replace(settings, mlflow_tracking_uri=uri))

    result = train_models(
        make_demo_data(600, seed=17),
        seed=17,
        artifact_dir=tmp_path / "artifacts",
        tune_lightgbm_trials=2,
        tuning_study_name=f"train-study-{uuid.uuid4().hex[:8]}",
        mlflow_tracking_uri=uri,
    )
    assert result["tuning"]["backend"] == "optuna"
    assert result["tuning"]["trials"] == 2
    assert result["tracking"]["backend"] == "mlflow", result["tracking"]["details"]
    assert result["tracking"]["details"]["model_logged"] is True
    assert (tmp_path / "artifacts" / "tuning_study.json").is_file()
