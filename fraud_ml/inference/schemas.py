"""Typed fraud scoring response objects."""
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Prediction:
    transaction_id: str
    fraud_probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW"]
    model_version: str

def classify_probability(transaction_id: str, probability: float, threshold: float, model_version: str) -> Prediction:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be between zero and one.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be between zero and one.")
    risk = "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.3 else "LOW"
    return Prediction(str(transaction_id), float(probability), risk,
                      "REVIEW" if probability >= threshold else "APPROVE", str(model_version))
