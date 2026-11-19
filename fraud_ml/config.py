"""Centralized, environment-backed application configuration."""
from dataclasses import dataclass
import os
from pathlib import Path


def _origins(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated CORS allowlist; never default to a wildcard."""
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or ("http://localhost:8501", "http://127.0.0.1:8501")


@dataclass(frozen=True)
class Settings:
    data_mode: str = os.getenv("FRAUD_DATA_MODE", "demo").lower()
    data_dir: Path = Path(os.getenv("FRAUD_DATA_DIR", "data"))
    artifact_path: Path = Path(os.getenv("FRAUD_MODEL_PATH", "artifacts/candidate.joblib"))
    decision_threshold: float = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    api_key: str = os.getenv("FRAUD_API_KEY", "")
    max_batch_size: int = int(os.getenv("FRAUD_MAX_BATCH_SIZE", "500"))
    random_seed: int = int(os.getenv("FRAUD_RANDOM_SEED", "42"))
    train_fraction: float = float(os.getenv("FRAUD_TRAIN_FRACTION", "0.70"))
    validation_fraction: float = float(os.getenv("FRAUD_VALIDATION_FRACTION", "0.15"))
    threshold_strategy: str = os.getenv("FRAUD_THRESHOLD_STRATEGY", "max_f1").lower()
    calibration_method: str = os.getenv("FRAUD_CALIBRATION_METHOD", "isotonic").lower()
    # MLflow 3.x requires a database-backed store; a file store is deprecated.
    mlflow_tracking_uri: str = os.getenv("FRAUD_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    cors_origins: tuple[str, ...] = _origins(os.getenv("FRAUD_CORS_ORIGINS", ""))
    api_url: str = os.getenv("FRAUD_API_URL", "http://localhost:8000")

    def validate(self) -> None:
        if self.data_mode not in {"real", "demo"}:
            raise ValueError("FRAUD_DATA_MODE must be 'real' or 'demo'.")
        if not 0 < self.decision_threshold < 1:
            raise ValueError("FRAUD_THRESHOLD must be between 0 and 1.")
        if self.train_fraction <= 0 or self.validation_fraction <= 0 or self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("Invalid partition fractions.")
        if self.threshold_strategy not in {"max_f1", "min_precision", "review_capacity", "business_cost"}:
            raise ValueError("Unsupported FRAUD_THRESHOLD_STRATEGY.")
        if self.calibration_method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError("FRAUD_CALIBRATION_METHOD must be none, sigmoid, or isotonic.")
        if self.max_batch_size < 1:
            raise ValueError("FRAUD_MAX_BATCH_SIZE must be at least 1.")
        if "*" in self.cors_origins:
            raise ValueError("Wildcard CORS origins are not permitted.")


settings = Settings()
settings.validate()
