"""Leakage-aware temporal model comparison and artifact persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_ml.data.splitting import chronological_split
from fraud_ml.features.engineering import engineer_features
from fraud_ml.features.pipeline import build_preprocessor
from fraud_ml.mlops.registry import evaluation_gates, register_candidate
from fraud_ml.mlops.tracking import code_revision, dataset_fingerprint, log_training_run
from fraud_ml.models.business import (
    BusinessAssumptions,
    evaluate_business,
    optimize_business_threshold,
)
from fraud_ml.models.calibration import fit_score_calibration
from fraud_ml.models.evaluate import evaluate_model
from fraud_ml.models.threshold import optimize_threshold

ThresholdStrategy = Literal["max_f1", "min_precision", "review_capacity", "business_cost"]


def train_models(
    df: pd.DataFrame,
    seed: int = 42,
    artifact_dir: str | Path = "artifacts",
    *,
    threshold_strategy: ThresholdStrategy = "max_f1",
    assumptions: BusinessAssumptions | None = None,
    tune_lightgbm_trials: int = 0,
    calibration_method: str = "isotonic",
    register: bool = True,
    experiment_name: str = "fraud-platform",
) -> dict:
    """Train, select, calibrate and persist models using temporal partitions.

    The final test partition is used once, after model selection, calibration and
    threshold choice have all been fixed on the validation partition.
    """
    assumptions = assumptions or BusinessAssumptions()
    assumptions.validate()
    parts = chronological_split(df)

    x_train = engineer_features(parts.train.drop(columns=["isFraud"], errors="ignore"))
    x_val = engineer_features(parts.validation.drop(columns=["isFraud"], errors="ignore"))
    x_test = engineer_features(parts.test.drop(columns=["isFraud"], errors="ignore"))
    y_train = parts.train.isFraud.astype(int)
    y_val = parts.validation.isFraud.astype(int)
    y_test = parts.test.isFraud.astype(int)
    amounts_val = pd.to_numeric(
        parts.validation.get("TransactionAmt", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    amounts_test = pd.to_numeric(
        parts.test.get("TransactionAmt", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    lightgbm_params: dict[str, Any] = {}
    try:
        from lightgbm import LGBMClassifier

        if tune_lightgbm_trials > 0:
            from fraud_ml.models.tuning import tune_lightgbm

            tuning = tune_lightgbm(
                x_train, y_train, x_val, y_val, n_trials=tune_lightgbm_trials, seed=seed
            )
            lightgbm_params = dict(tuning.best_params)
        lightgbm_params.setdefault("n_estimators", 350)
        lightgbm_params.setdefault("learning_rate", 0.05)
        lightgbm_params.setdefault("num_leaves", 31)
        lightgbm_params.setdefault("verbosity", -1)
        models["LightGBM"] = LGBMClassifier(
            **lightgbm_params,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    except ImportError:
        pass

    results: dict[str, dict[str, Any]] = {}
    for name, estimator in models.items():
        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(scale_numeric=name == "Logistic Regression"),
                ),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)

        raw_val_probability = pipeline.predict_proba(x_val)[:, 1]
        calibrator, calibration_report = fit_score_calibration(
            y_val, raw_val_probability, method=calibration_method, seed=seed
        )
        val_probability = calibrator.transform(raw_val_probability)

        if threshold_strategy == "business_cost":
            selected = optimize_business_threshold(
                y_val, val_probability, amounts_val, assumptions
            )
            cutoff = float(selected["threshold"])
            threshold_details = selected
        else:
            selected_threshold = optimize_threshold(
                y_val,
                val_probability,
                strategy=threshold_strategy,
                min_precision=0.5,
                capacity=max(1, int(len(y_val) * 0.05)),
                false_positive_cost=assumptions.false_positive_cost,
                missed_fraud_cost=assumptions.missed_fraud_cost,
            )
            cutoff = float(selected_threshold["threshold"])
            threshold_details = selected_threshold

        raw_test_probability = pipeline.predict_proba(x_test)[:, 1]
        test_probability = calibrator.transform(raw_test_probability)
        validation_metrics = evaluate_model(y_val, val_probability, cutoff)
        test_metrics = evaluate_model(y_test, test_probability, cutoff)
        validation_business = evaluate_business(
            y_val, val_probability, amounts_val, cutoff, assumptions
        )
        test_business = evaluate_business(
            y_test, test_probability, amounts_test, cutoff, assumptions
        )

        results[name] = {
            "pipeline": pipeline,
            "calibrator": calibrator,
            "threshold": cutoff,
            "calibration": calibration_report,
            "threshold_details": threshold_details,
            "validation": validation_metrics,
            "test": test_metrics,
            "validation_business": validation_business,
            "test_business": test_business,
            "validation_probability": val_probability,
            "test_probability": test_probability,
        }

    if not results:
        raise RuntimeError("No model candidates available.")

    champion = max(results, key=lambda key: results[key]["validation"]["average_precision"])
    best = results[champion]

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_fingerprint(df)
    revision = code_revision()
    schema = {
        "columns": list(x_train.columns),
        "dtypes": {str(column): str(dtype) for column, dtype in x_train.dtypes.items()},
    }
    bundle = {
        "pipeline": best["pipeline"],
        "calibrator": best["calibrator"],
        "calibration": best["calibration"],
        "model_name": champion,
        "threshold": best["threshold"],
        "threshold_strategy": threshold_strategy,
        "business_assumptions": assumptions.__dict__,
        "feature_columns": list(x_train.columns),
        "feature_schema": schema,
        "dataset_fingerprint": fingerprint,
        "code_revision": revision,
        "training_config": {
            "seed": seed,
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "threshold_strategy": threshold_strategy,
            "calibration_method": calibration_method,
            "lightgbm_params": lightgbm_params,
        },
        "validation_metrics": best["validation"],
        "test_metrics": best["test"],
        "artifact_version": "1.0.0",
    }
    artifact_path = output / "candidate.joblib"
    import joblib

    joblib.dump(bundle, artifact_path)

    report = {
        name: {
            "threshold": item["threshold"],
            "threshold_details": item["threshold_details"],
            "calibration": item["calibration"],
            "validation": item["validation"],
            "test": item["test"],
            "validation_business": item["validation_business"],
            "test_business": item["test_business"],
        }
        for name, item in results.items()
    }
    report_payload = {
        "selection_metric": "validation average_precision",
        "candidate": champion,
        "dataset_fingerprint": fingerprint,
        "code_revision": revision,
        "final_test_used_once": True,
        "models": report,
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(report_payload, indent=2, default=str), encoding="utf-8")

    gate = evaluation_gates(best["validation"])
    if register:
        register_candidate(
            artifact_path,
            model_name=champion,
            version="1.0.0",
            validation_metrics=best["validation"],
            gate=gate,
            registry_dir=output / "registry",
        )

    tracking = log_training_run(
        experiment=experiment_name,
        run_name=f"{champion}-{fingerprint[:8]}",
        parameters=bundle["training_config"],
        metrics={
            "validation_average_precision": best["validation"]["average_precision"],
            "validation_roc_auc": best["validation"]["roc_auc"] or 0.0,
            "validation_brier_score": best["validation"]["brier_score"],
            "test_average_precision": best["test"]["average_precision"],
            "test_roc_auc": best["test"]["roc_auc"] or 0.0,
        },
        tags={
            "dataset_fingerprint": fingerprint,
            "code_revision": revision,
            "model_name": champion,
            "threshold_strategy": threshold_strategy,
            "calibration_method": calibration_method,
        },
        artifact_paths=[artifact_path, metrics_path],
        output_dir=output / "ml_tracking",
    )

    return {
        "results": results,
        "best_name": champion,
        "artifact_path": str(artifact_path),
        "report": report,
        "fingerprint": fingerprint,
        "code_revision": revision,
        "gates": gate.as_dict(),
        "tracking": tracking.as_dict(),
    }
