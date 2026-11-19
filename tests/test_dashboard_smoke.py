"""Smoke/regression tests for the preserved Streamlit data and ML modules.

The copilot branch dropped this file; it is restored here against the explicit
demo generator so legacy dashboard helpers remain covered. Synthetic records
are used for software correctness only, never as performance evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_ml.data.ingestion import make_demo_data
from src.data_loader import get_class_distribution
from src.model_trainer import train_all
from src.preprocessing import engineer_features, preprocess


def test_demo_data_is_reproducible_and_has_expected_schema() -> None:
    first = make_demo_data(rows=800, seed=7)
    second = make_demo_data(rows=800, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 800
    assert {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"}.issubset(first.columns)
    assert set(first["isFraud"].unique()) == {0, 1}
    assert first["TransactionID"].is_unique
    assert (first["TransactionAmt"] >= 0).all()


def test_class_distribution_counts_both_classes() -> None:
    data = pd.DataFrame({"isFraud": [0, 0, 1, 0, 1]})
    assert get_class_distribution(data) == {"Legitimate": 3, "Fraud": 2}


def test_feature_engineering_adds_time_and_amount_features() -> None:
    data = pd.DataFrame({"TransactionDT": [0, 3600, 23 * 3600], "TransactionAmt": [0.0, 10.0, 2.5]})

    result = engineer_features(data)

    assert result["hour"].tolist() == [0, 1, 23]
    assert result["is_night"].tolist() == [1, 1, 1]
    assert result["log_amt"].iloc[1] == pytest.approx(np.log1p(10.0))
    assert result["amt_decimal"].iloc[2] == pytest.approx(0.5)
    assert "hour" not in data.columns


def test_preprocessing_fits_and_reuses_encoders_with_unseen_categories() -> None:
    data = make_demo_data(rows=800, seed=21)
    x_train, y_train, encoders, columns = preprocess(data, fit=True)

    assert x_train.shape == (800, len(columns))
    assert y_train.shape == (800,)
    assert np.isfinite(x_train).all()

    inference = data.drop(columns=["isFraud"]).iloc[:3].copy()
    inference.loc[inference.index[0], "card4"] = "new-card-type"

    for column in encoders:
        if column == "_numeric_medians" or column not in inference.columns:
            continue
        encoder = encoders[column]
        inference[column] = inference[column].fillna("unknown").astype(str)
        inference[column] = inference[column].apply(
            lambda value: value if value in set(encoder.classes_) else "unknown"
        )

    x_inference, y_inference, _, inference_columns = preprocess(
        inference, fit=False, encoders=encoders
    )
    assert inference_columns == columns
    assert x_inference.shape == (3, len(columns))
    assert y_inference is None
    assert np.isfinite(x_inference).all()


def test_logistic_training_smoke_returns_metrics_and_artifact() -> None:
    data = make_demo_data(rows=1000, seed=42)
    x, y, _, feature_names = preprocess(data, fit=True)

    results, x_test, y_test = train_all(
        x, y, feature_names, use_lgb=False, use_rf=False, use_lr=True
    )

    assert list(results) == ["Logistic Regression"]
    result = results["Logistic Regression"]
    assert result["model"] is not None
    assert result["scaler"] is not None
    assert x_test.shape[0] == y_test.shape[0]
    assert 0.0 <= result["metrics"]["roc_auc"] <= 1.0
    assert 0.0 <= result["metrics"]["avg_prec"] <= 1.0
    assert "confusion_matrix" in result["metrics"]


def test_training_rejects_no_selected_models() -> None:
    data = make_demo_data(rows=200, seed=9)
    x, y, _, feature_names = preprocess(data, fit=True)

    with pytest.raises(ValueError, match="No models selected"):
        train_all(x, y, feature_names, use_lgb=False, use_rf=False, use_lr=False)
