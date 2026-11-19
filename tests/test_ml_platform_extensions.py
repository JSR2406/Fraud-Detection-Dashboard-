"""Tests for calibration, business evaluation, explanations, registry and monitoring.

Synthetic data only: these tests verify software correctness, not fraud performance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_ml.data.ingestion import make_demo_data
from fraud_ml.features.engineering import engineer_features
from fraud_ml.inference.predictor import FraudPredictor
from fraud_ml.mlops.monitoring import build_reference_profile, compare_to_reference, write_monitoring_report
from fraud_ml.mlops.registry import (
    evaluation_gates,
    load_alias,
    promote,
    register_candidate,
    rollback,
)
from fraud_ml.mlops.tracking import code_revision, dataset_fingerprint
from fraud_ml.models.business import BusinessAssumptions, evaluate_business, optimize_business_threshold
from fraud_ml.models.calibration import ScoreCalibrator, fit_score_calibration
from fraud_ml.models.explain import explain_transactions, explanation_support
from fraud_ml.models.threshold import optimize_threshold


def test_threshold_is_never_degenerate() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.25, 0.75, 0.95])
    for strategy, kwargs in (
        ("max_f1", {}),
        ("min_precision", {"min_precision": 0.5}),
        ("review_capacity", {"capacity": 2}),
        ("business_cost", {}),
    ):
        result = optimize_threshold(y, p, strategy=strategy, **kwargs)
        assert 0.0 < result["threshold"] < 1.0


def test_score_calibration_reports_brier_and_transforms() -> None:
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 400)
    p = np.clip(rng.beta(1.5, 4.0, 400) + y * 0.25, 1e-6, 1 - 1e-6)
    calibrator, report = fit_score_calibration(y, p, method="isotonic", seed=7)
    assert 0.0 <= report["brier_before"] <= 1.0
    assert 0.0 <= report["brier_after"] <= 1.0
    assert isinstance(calibrator, ScoreCalibrator)
    transformed = calibrator.transform([0.1, 0.5, 0.9])
    assert transformed.shape == (3,)
    assert ((transformed >= 0) & (transformed <= 1)).all()


def test_business_evaluation_uses_explicit_assumptions() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.8, 0.9, 0.2])
    amounts = np.array([100.0, 200.0, 500.0, 400.0])
    assumptions = BusinessAssumptions(
        false_positive_cost=10, missed_fraud_cost=100, amount_at_risk_weight=0.1,
        investigation_capacity=2,
    )
    result = evaluate_business(y, p, amounts, 0.5, assumptions)
    assert result["missed_fraud_amount"] == pytest.approx(400.0)
    assert result["estimated_total_cost"] > 0
    assert result["fraud_capture_rate"] == pytest.approx(0.5)
    assert "not realized savings" in result["disclaimer"]
    best = optimize_business_threshold(y, p, amounts, assumptions)
    assert 0.0 < best["threshold"] < 1.0
    with pytest.raises(ValueError, match="non-negative"):
        BusinessAssumptions(false_positive_cost=-1).validate()


def test_explanations_degrade_gracefully_when_shap_unavailable() -> None:
    available, reason = explanation_support()
    assert isinstance(available, bool)
    assert isinstance(reason, str)
    result = explain_transactions(object(), pd.DataFrame(), pd.DataFrame())
    assert isinstance(result.available, bool)
    assert result.as_dict()["available"] == result.available


def test_registry_gates_promotion_and_rollback(tmp_path: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from fraud_ml.features.pipeline import build_preprocessor

    raw = make_demo_data(400, seed=5)
    features = engineer_features(raw.drop(columns="isFraud"))
    pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", LogisticRegression(max_iter=300))])
    pipeline.fit(features, raw["isFraud"])
    artifact = tmp_path / "model.joblib"
    import joblib

    joblib.dump(
        {
            "pipeline": pipeline,
            "model_name": "test",
            "threshold": 0.4,
            "feature_columns": list(features.columns),
            "artifact_version": "test-v1",
        },
        artifact,
    )
    good = evaluation_gates({"average_precision": 0.6, "brier_score": 0.1, "false_positive_rate": 0.2})
    assert good.passed
    bad = evaluation_gates({"average_precision": 0.01, "brier_score": 0.9, "false_positive_rate": 0.99})
    assert not bad.passed and len(bad.failures) == 3

    registry_dir = tmp_path / "registry"
    register_candidate(artifact, model_name="test", version="1", validation_metrics={"average_precision": 0.6}, gate=good, registry_dir=registry_dir)
    register_candidate(artifact, model_name="test", version="2", validation_metrics={"average_precision": 0.7}, gate=good, registry_dir=registry_dir)
    promoted = promote("1", registry_dir=registry_dir)
    assert promoted["previous"] is None
    assert isinstance(load_alias(registry_dir=registry_dir), FraudPredictor)
    promote("2", registry_dir=registry_dir)
    rolled = rollback(registry_dir=registry_dir)
    assert rolled["version"] == "1"


def test_monitoring_profile_and_drift_report(tmp_path: Path) -> None:
    reference = make_demo_data(500, seed=11)
    profile = build_reference_profile(reference)
    assert profile["rows"] == 500
    report = compare_to_reference(reference, profile)
    assert report["status"] in {"ok", "alert"}
    path = write_monitoring_report(report, tmp_path / "monitoring.json")
    assert json.loads(path.read_text(encoding="utf-8"))["rows"] == 500


def test_training_bundle_includes_calibration_and_metadata(tmp_path: Path) -> None:
    from fraud_ml.models.train import train_models

    result = train_models(make_demo_data(900, seed=23), seed=23, artifact_dir=tmp_path)
    assert Path(result["artifact_path"]).is_file()
    assert result["gates"] is not None
    assert result["tracking"]["backend"] in {"mlflow", "local-json-fallback"}
    predictor = FraudPredictor.from_trusted_file(result["artifact_path"])
    assert predictor.calibration["method"].startswith("isotonic")
    row = {"transaction_id": "cal-001", "TransactionAmt": 42.0, "ProductCD": "unknown-category"}
    scored = predictor.predict([row])[0]
    assert 0.0 <= scored["fraud_probability"] <= 1.0
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["final_test_used_once"] is True
    assert dataset_fingerprint(make_demo_data(10)) == dataset_fingerprint(make_demo_data(10))
    assert isinstance(code_revision(), str)
