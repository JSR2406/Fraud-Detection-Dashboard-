"""Business-oriented fraud evaluation with explicit, auditable assumptions.

All monetary figures are *estimates derived from operator-supplied assumptions*.
They are not measured savings and must never be reported as realized financial
impact without real loss data and a validated cost model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class BusinessAssumptions:
    """Cost and capacity assumptions for a review operation."""

    false_positive_cost: float = 50.0
    missed_fraud_cost: float = 500.0
    amount_at_risk_weight: float = 1.0
    investigation_capacity: int | None = None

    def validate(self) -> None:
        if self.false_positive_cost < 0 or self.missed_fraud_cost < 0:
            raise ValueError("Cost assumptions must be non-negative.")
        if self.amount_at_risk_weight < 0:
            raise ValueError("amount_at_risk_weight must be non-negative.")
        if self.investigation_capacity is not None and self.investigation_capacity < 1:
            raise ValueError("investigation_capacity must be at least 1.")


def evaluate_business(
    y_true: np.ndarray | list[int],
    probability: np.ndarray | list[float],
    transaction_amounts: np.ndarray | list[float] | None,
    threshold: float,
    assumptions: BusinessAssumptions | None = None,
) -> dict:
    """Compute assumption-based costs, amount-at-risk, and review capacity KPIs."""
    assumptions = assumptions or BusinessAssumptions()
    assumptions.validate()
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid labels or probabilities.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one.")

    amounts = (
        np.asarray(transaction_amounts, dtype=float)
        if transaction_amounts is not None
        else np.zeros(len(y), dtype=float)
    )
    if len(amounts) != len(y) or not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("transaction_amounts must be finite, non-negative, and aligned.")
    amounts = np.clip(amounts, 0.0, None)

    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()

    missed_amount = float(amounts[(y == 1) & (predicted == 0)].sum())
    flagged_amount = float(amounts[predicted == 1].sum())
    estimated_cost = (
        int(fp) * assumptions.false_positive_cost
        + int(fn) * assumptions.missed_fraud_cost
        + missed_amount * assumptions.amount_at_risk_weight
    )

    capacity = assumptions.investigation_capacity
    if capacity is None:
        capacity = int(fp + tp)
    capacity = min(max(int(capacity), 1), len(y))
    review_index = np.argsort(p)[-capacity:]
    reviewed_fraud = int(y[review_index].sum())
    total_fraud = max(int(y.sum()), 1)

    return {
        "threshold": float(threshold),
        "estimated_investigation_cost": float(int(fp) * assumptions.false_positive_cost),
        "estimated_missed_fraud_cost": float(int(fn) * assumptions.missed_fraud_cost),
        "estimated_amount_at_risk_cost": float(missed_amount * assumptions.amount_at_risk_weight),
        "estimated_total_cost": float(estimated_cost),
        "missed_fraud_amount": missed_amount,
        "flagged_transaction_amount": flagged_amount,
        "fraud_capture_rate": float(reviewed_fraud / total_fraud),
        "review_capacity": capacity,
        "assumptions": asdict(assumptions),
        "disclaimer": (
            "Costs are estimates from configurable assumptions, not realized savings. "
            "Replace with validated loss data before financial claims."
        ),
    }


def optimize_business_threshold(
    y_true: np.ndarray | list[int],
    probability: np.ndarray | list[float],
    transaction_amounts: np.ndarray | list[float] | None,
    assumptions: BusinessAssumptions | None = None,
) -> dict:
    """Choose the validation threshold minimizing estimated business cost."""
    assumptions = assumptions or BusinessAssumptions()
    assumptions.validate()
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("Invalid threshold inputs.")
    lower = float(np.nextafter(0.0, 1.0))
    upper = float(np.nextafter(1.0, 0.0))
    grid = np.unique(np.r_[lower, p[(p > lower) & (p < upper)], upper])

    best: dict | None = None
    for threshold in grid:
        metrics = evaluate_business(y, p, transaction_amounts, float(threshold), assumptions)
        if best is None or metrics["estimated_total_cost"] < best["estimated_total_cost"]:
            best = metrics
    if best is None:  # pragma: no cover - grid is never empty for valid input
        raise ValueError("No threshold candidates available.")
    return best
