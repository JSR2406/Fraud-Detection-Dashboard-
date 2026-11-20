"""Real-backend tests for Pandera schema enforcement and native SHAP."""
from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from fraud_ml.data.ingestion import load_dataset, make_demo_data
from fraud_ml.data.validation import pandera_available, validate_dataset
from fraud_ml.features.engineering import engineer_features
from fraud_ml.features.pipeline import build_preprocessor
from fraud_ml.models.explain import explanation_support, explain_transactions


def test_pandera_schema_is_enforced_on_loaded_data() -> None:
    installed, detail = pandera_available()
    if not installed:
        pytest.skip(f"pandera unavailable: {detail}")

    frame, _ = load_dataset("demo", sample_rows=300, use_pandera=True)
    assert len(frame) == 300

    bad_amount = frame.copy()
    bad_amount.loc[0, "TransactionAmt"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        validate_dataset(bad_amount, use_pandera=True)

    bad_label = frame.copy()
    bad_label.loc[0, "isFraud"] = 7
    with pytest.raises(ValueError):
        validate_dataset(bad_label, use_pandera=True)


def test_native_shap_produces_bounded_explanations() -> None:
    available, reason = explanation_support()
    if not available:
        pytest.skip(reason or "native SHAP unavailable")

    raw = make_demo_data(400, seed=9)
    features = engineer_features(raw)
    model = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1)),
        ]
    )
    model.fit(features, raw.isFraud.astype(int))

    result = explain_transactions(
        model,
        features.head(60),
        features.head(3),
        max_background=20,
        max_explain=2,
        top_features=4,
    )
    assert result.available, result.reason
    assert len(result.local_explanations) == 2
    assert len(result.global_importance) == 4
    assert all(item["mean_abs_shap"] >= 0 for item in result.global_importance)
