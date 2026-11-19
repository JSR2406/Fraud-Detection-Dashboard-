"""Leakage-aware temporal model comparison and artifact persistence."""
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from fraud_ml.data.splitting import chronological_split
from fraud_ml.features.engineering import engineer_features
from fraud_ml.features.pipeline import build_preprocessor
from fraud_ml.models.evaluate import evaluate_model
from fraud_ml.models.threshold import optimize_threshold


def train_models(df: pd.DataFrame, seed: int = 42, artifact_dir: str | Path = "artifacts") -> dict:
    parts = chronological_split(df)
    x_train = engineer_features(parts.train.drop(columns=["isFraud"], errors="ignore"))
    x_val = engineer_features(parts.validation.drop(columns=["isFraud"], errors="ignore"))
    x_test = engineer_features(parts.test.drop(columns=["isFraud"], errors="ignore"))
    y_train, y_val, y_test = (parts.train.isFraud.astype(int), parts.validation.isFraud.astype(int), parts.test.isFraud.astype(int))
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=160, max_depth=14, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
    }
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(n_estimators=350, learning_rate=.05, num_leaves=31, class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1)
    except ImportError:
        pass
    results = {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(scale_numeric=name == "Logistic Regression")), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        val_probs = pipeline.predict_proba(x_val)[:, 1]
        cutoff = optimize_threshold(y_val.to_numpy(), val_probs)["threshold"]
        test_probs = pipeline.predict_proba(x_test)[:, 1]
        results[name] = {"pipeline": pipeline, "threshold": cutoff,
                         "validation": evaluate_model(y_val, val_probs, cutoff),
                         "test": evaluate_model(y_test, test_probs, cutoff),
                         "validation_probability": val_probs, "test_probability": test_probs}
    if not results:
        raise RuntimeError("No model candidates available.")
    champion = max(results, key=lambda key: results[key]["validation"]["average_precision"])
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": results[champion]["pipeline"], "model_name": champion,
              "threshold": results[champion]["threshold"], "feature_columns": list(x_train.columns),
              "artifact_version": "1.0.0", "calibration": "uncalibrated"}
    import joblib
    joblib.dump(bundle, output / "candidate.joblib")
    report = {name: {"threshold": item["threshold"], "validation": item["validation"], "test": item["test"]} for name, item in results.items()}
    (output / "metrics.json").write_text(json.dumps({"selection_metric": "validation average_precision", "candidate": champion, "models": report}, indent=2), encoding="utf-8")
    return {"results": results, "best_name": champion, "artifact_path": str(output / "candidate.joblib"), "report": report}
