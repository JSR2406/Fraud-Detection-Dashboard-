"""Strict predictor for trusted persisted pipeline bundles."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fraud_ml.features.engineering import RAW_COLUMNS, engineer_features
from fraud_ml.inference.schemas import classify_probability

class FraudPredictor:
    def __init__(self, bundle: dict):
        required = {"pipeline", "model_name", "threshold"}
        if not isinstance(bundle, dict) or not required.issubset(bundle):
            raise ValueError("Unsupported fraud model artifact schema.")
        self.pipeline = bundle["pipeline"]
        self.model_name = str(bundle["model_name"])
        self.threshold = float(bundle["threshold"])
        self.model_version = str(bundle.get("artifact_version", "unknown"))
        self.feature_columns = bundle.get("feature_columns")
        self.calibrator = bundle.get("calibrator")
        self.calibration = bundle.get("calibration", {"method": "uncalibrated"})
        if not 0 < self.threshold < 1:
            raise ValueError("Stored model threshold must be between zero and one.")

    @classmethod
    def from_trusted_file(cls, path: str | Path) -> "FraudPredictor":
        """Load only trusted operator-provisioned artifacts; joblib deserialization is executable."""
        artifact = Path(path).resolve()
        if not artifact.is_file():
            raise FileNotFoundError(f"Model artifact not found: {artifact}")
        return cls(joblib.load(artifact))

    def predict(self, rows: list[dict], threshold: float | None = None) -> list[dict]:
        frame = pd.DataFrame(rows)
        if frame.empty or "transaction_id" not in frame:
            raise ValueError("Provide at least one row with transaction_id.")
        ids = frame.pop("transaction_id").astype(str).tolist()
        unknown = set(frame.columns) - set(RAW_COLUMNS)
        if unknown:
            raise ValueError(f"Unexpected feature fields: {sorted(unknown)}")
        if "TransactionAmt" not in frame:
            raise ValueError("TransactionAmt is required.")
        amounts = pd.to_numeric(frame["TransactionAmt"], errors="coerce")
        if amounts.isna().any() or (amounts < 0).any():
            raise ValueError("TransactionAmt must be finite and non-negative.")
        frame["TransactionAmt"] = amounts
        features = engineer_features(frame)
        if self.feature_columns is not None and list(features.columns) != list(self.feature_columns):
            raise ValueError("Feature schema/order does not match model artifact.")
        cutoff = self.threshold if threshold is None else float(threshold)
        if not 0 < cutoff < 1:
            raise ValueError("Threshold must be between zero and one.")
        probabilities = np.asarray(self.pipeline.predict_proba(features)[:, 1], dtype=float)
        if self.calibrator is not None:
            probabilities = np.asarray(self.calibrator.transform(probabilities), dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError("Model produced non-finite fraud probabilities.")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return [classify_probability(txid, float(prob), cutoff, self.model_version).__dict__
                for txid, prob in zip(ids, probabilities)]
