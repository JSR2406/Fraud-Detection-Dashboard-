"""End-to-end regression suite for the fraud ML path.

These tests exercise synthetic examples only. Their purpose is software correctness,
not model-quality claims or evidence of real-world fraud performance.
"""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_ml.data.ingestion import load_dataset, load_ieee_cis, make_demo_data
from fraud_ml.data.splitting import chronological_split
from fraud_ml.data.validation import quality_report, validate_dataset
from fraud_ml.features.engineering import CATEGORICAL, NUMERIC, engineer_features
from fraud_ml.features.pipeline import build_preprocessor
from fraud_ml.inference.predictor import FraudPredictor
from fraud_ml.models.evaluate import evaluate_model
from fraud_ml.models.threshold import optimize_threshold


def test_demo_mode_is_explicit_reproducible_and_labeled() -> None:
    first, source = load_dataset("demo", seed=9, sample_rows=600)
    second, _ = load_dataset("demo", seed=9, sample_rows=600)
    assert "synthetic" in source
    pd.testing.assert_frame_equal(first, second)
    with pytest.raises(ValueError, match="explicitly"):
        load_dataset("unknown-mode")


def test_real_mode_fails_clearly_instead_of_falling_back(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="transaction file"):
        load_dataset("real", tmp_path)


def test_csv_zip_identity_join_and_quality_report(tmp_path: Path) -> None:
    tx = pd.DataFrame({"TransactionID": [1, 2], "TransactionAmt": [10., 20.], "isFraud": [0, 1]})
    identity = pd.DataFrame({"TransactionID": [1], "DeviceType": ["mobile"]})
    tx.to_csv(tmp_path / "train_transaction.csv", index=False)
    with zipfile.ZipFile(tmp_path / "train_identity.csv.zip", "w") as archive:
        archive.writestr("train_identity.csv", identity.to_csv(index=False))
    joined = load_ieee_cis(tmp_path)
    assert joined.loc[0, "DeviceType"] == "mobile"
    assert pd.isna(joined.loc[1, "DeviceType"])
    quality = quality_report(joined)
    assert quality["rows"] == 2 and quality["fraud_count"] == 1
    assert quality["missing_cells"] == 1


def test_validation_rejects_duplicate_ids_bad_amount_and_target() -> None:
    data = make_demo_data(300)
    duplicate = data.copy()
    duplicate.loc[1, "TransactionID"] = duplicate.loc[0, "TransactionID"]
    with pytest.raises(ValueError, match="unique"):
        validate_dataset(duplicate)
    negative = data.copy()
    negative.loc[0, "TransactionAmt"] = -0.01
    with pytest.raises(ValueError, match="TransactionAmt"):
        validate_dataset(negative)
    nonfinite = data.copy()
    nonfinite.loc[0, "TransactionAmt"] = np.inf
    with pytest.raises(ValueError, match="TransactionAmt"):
        validate_dataset(nonfinite)
    invalid_target = data.copy()
    invalid_target.loc[0, "isFraud"] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_dataset(invalid_target)


def test_chronological_split_orders_data_and_rejects_invalid_partitions() -> None:
    data = make_demo_data(1200, seed=3).sample(frac=1, random_state=10)
    parts = chronological_split(data)
    assert parts.train.TransactionDT.max() <= parts.validation.TransactionDT.min()
    assert parts.validation.TransactionDT.max() <= parts.test.TransactionDT.min()
    assert len(parts.train) + len(parts.validation) + len(parts.test) == len(data)
    with pytest.raises(ValueError, match="TransactionDT"):
        chronological_split(data.drop(columns="TransactionDT"))
    with pytest.raises(ValueError, match="fractions"):
        chronological_split(data, train_fraction=.9, validation_fraction=.2)


def test_feature_engineering_keeps_schema_and_handles_unseen_values() -> None:
    frame = pd.DataFrame([{"TransactionAmt": 10.25, "TransactionDT": 36000,
                           "ProductCD": None, "card4": "new-issuer"}])
    features = engineer_features(frame)
    assert features.columns.tolist() == NUMERIC + CATEGORICAL
    assert features.loc[0, "card4"] == "new-issuer"
    assert features.loc[0, "ProductCD"] == "unknown"
    assert features.loc[0, "log_amt"] == pytest.approx(np.log1p(10.25))
    assert features.loc[0, "hour"] == 10


def test_preprocessor_fits_and_transforms_unseen_category() -> None:
    train = engineer_features(make_demo_data(500, seed=12))
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(train)
    unseen = train.iloc[[0]].copy()
    unseen.loc[:, "card4"] = "category-never-seen"
    output = preprocessor.transform(unseen)
    assert output.shape[1] == transformed.shape[1]
    assert np.isfinite(output.toarray() if hasattr(output, "toarray") else output).all()


def test_metrics_and_threshold_selection_are_consistent() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([.05, .25, .75, .95])
    threshold = optimize_threshold(y, probability, strategy="max_f1")
    metrics = evaluate_model(y, probability, threshold["threshold"], k=2)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["precision_at_k"] == pytest.approx(1.0)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["brier_score"] < .1
    with pytest.raises(ValueError, match="capacity"):
        optimize_threshold(y, probability, strategy="review_capacity")
    capacity = optimize_threshold(y, probability, strategy="review_capacity", capacity=1)
    assert capacity["objective"] == 1


def test_train_persist_reload_predict_end_to_end(tmp_path: Path) -> None:
    """Train on a temporal demo split, persist/load, then score unseen categories."""
    from fraud_ml.models.train import train_models

    data = make_demo_data(1800, seed=42)
    output = train_models(data, seed=42, artifact_dir=tmp_path)
    assert Path(output["artifact_path"]).is_file()
    metrics_file = tmp_path / "metrics.json"
    report = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert report["candidate"] == output["best_name"]
    assert report["selection_metric"] == "validation average_precision"
    assert "test" in report["models"][output["best_name"]]

    predictor = FraudPredictor.from_trusted_file(output["artifact_path"])
    response = predictor.predict([{"transaction_id": "e2e-001", "TransactionAmt": 52.3,
                                   "TransactionDT": 999999, "ProductCD": "never-seen-product",
                                   "card4": "never-seen-issuer", "DeviceType": "tablet"}])[0]
    assert response["transaction_id"] == "e2e-001"
    assert 0.0 <= response["fraud_probability"] <= 1.0
    assert response["decision"] in {"APPROVE", "REVIEW"}
    assert response["model_version"] == "1.0.0"


def test_predictor_rejects_bad_schema_and_invalid_amount() -> None:
    # Use a small fitted pipeline so the contract can be tested independently of training.
    raw = make_demo_data(400, seed=17)
    x = engineer_features(raw.drop(columns="isFraud"))
    pipeline = Pipeline([("preprocessor", build_preprocessor()),
                         ("model", LogisticRegression(max_iter=300, class_weight="balanced"))])
    pipeline.fit(x, raw["isFraud"])
    predictor = FraudPredictor({"pipeline": pipeline, "threshold": .4, "model_name": "test",
                                "feature_columns": list(x.columns), "artifact_version": "test-v1"})
    with pytest.raises(ValueError, match="Unexpected"):
        predictor.predict([{"transaction_id": "bad", "TransactionAmt": 2, "private_extra": 1}])
    with pytest.raises(ValueError, match="TransactionAmt"):
        predictor.predict([{"transaction_id": "bad", "TransactionAmt": -2}])
