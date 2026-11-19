"""Centralized, environment-backed application configuration."""
from dataclasses import dataclass
import os
from pathlib import Path
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
    def validate(self) -> None:
        if self.data_mode not in {"real", "demo"}: raise ValueError("FRAUD_DATA_MODE must be 'real' or 'demo'.")
        if not 0 < self.decision_threshold < 1: raise ValueError("FRAUD_THRESHOLD must be between 0 and 1.")
        if self.train_fraction <= 0 or self.validation_fraction <= 0 or self.train_fraction + self.validation_fraction >= 1: raise ValueError("Invalid partition fractions.")
settings = Settings()
