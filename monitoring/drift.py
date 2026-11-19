"""Descriptive data-quality and drift summaries.

These helpers use only pandas/numpy and the project's own validation module.
They report simple deltas against a baseline snapshot; they are **not**
statistical drift tests, concept-drift detectors, or model-performance
monitoring, and they require no external services.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd

from fraud_ml.data.validation import quality_report

AMOUNT_COLUMN = "TransactionAmt"
TARGET_COLUMN = "isFraud"


def _finite_numbers(series: pd.Series) -> pd.Series:
    """Coerce a column to finite numeric values, dropping NaN/inf entries."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna()


def amount_statistics(
    frame: pd.DataFrame, column: str = AMOUNT_COLUMN
) -> dict[str, float | int | None]:
    """Summarise the transaction-amount distribution as JSON-safe values."""
    empty: dict[str, float | int | None] = {
        "count": 0,
        "mean": None,
        "std": None,
        "min": None,
        "p50": None,
        "p95": None,
        "p99": None,
        "max": None,
        "zero_fraction": None,
    }
    if column not in frame.columns:
        return empty
    values = _finite_numbers(frame[column])
    if values.empty:
        return empty
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
        "zero_fraction": float((values == 0).mean()),
    }


def quality_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    """Return project quality metrics plus per-column missing fractions."""
    report = dict(quality_report(frame))
    report["missing_fraction_by_column"] = {
        str(column): float(frame[column].isna().mean()) if len(frame) else 0.0
        for column in frame.columns
    }
    return report


def fraud_statistics(
    frame: pd.DataFrame, column: str = TARGET_COLUMN
) -> dict[str, int | float | None]:
    """Return fraud count and rate, or nulls when the label is unavailable."""
    if column not in frame.columns:
        return {"count": None, "rate": None}
    target = pd.to_numeric(frame[column], errors="coerce").dropna()
    if target.empty:
        return {"count": 0, "rate": None}
    return {"count": int(target.sum()), "rate": float(target.mean())}


def create_snapshot(frame: pd.DataFrame, source: str) -> dict[str, Any]:
    """Build one timestamped, JSON-serialisable snapshot of a dataset."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "rows": int(len(frame)),
        "quality": quality_statistics(frame),
        "amount": amount_statistics(frame),
        "fraud": fraud_statistics(frame),
    }


def _delta(current: Any, baseline: Any) -> float | None:
    if current is None or baseline is None:
        return None
    try:
        return float(current) - float(baseline)
    except (TypeError, ValueError):
        return None


def compare_snapshots(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    rate_threshold: float = 0.05,
    amount_threshold: float = 0.20,
) -> dict[str, Any]:
    """Compare a small set of descriptive statistics against a baseline.

    ``rate_threshold`` is an absolute delta for missing/fraud rates;
    ``amount_threshold`` is a relative delta for amount mean/p95.
    """
    if rate_threshold < 0 or amount_threshold < 0:
        raise ValueError("Drift thresholds must be non-negative.")

    specs = (
        ("missing_fraction", "quality", rate_threshold, "absolute"),
        ("fraud_rate", "fraud", rate_threshold, "absolute"),
        ("mean", "amount", amount_threshold, "relative"),
        ("p95", "amount", amount_threshold, "relative"),
    )
    metrics: dict[str, Any] = {}
    for name, section, threshold, mode in specs:
        baseline_value = (baseline.get(section) or {}).get(name)
        current_value = (current.get(section) or {}).get(name)
        delta = _delta(current_value, baseline_value)
        if delta is None:
            observed: float | None = None
        elif mode == "relative" and baseline_value not in (None, 0):
            observed = abs(delta / float(baseline_value))
        else:
            observed = abs(delta)
        metrics[name] = {
            "baseline": baseline_value,
            "current": current_value,
            "delta": delta,
            "comparison": mode,
            "observed": observed,
            "threshold": float(threshold),
            "exceeded": bool(observed is not None and observed > threshold),
        }
    return {
        "thresholds": {
            "rate_absolute": float(rate_threshold),
            "amount_relative": float(amount_threshold),
        },
        "metrics": metrics,
        "any_exceeded": any(item["exceeded"] for item in metrics.values()),
    }


def create_report(
    frame: pd.DataFrame,
    source: str,
    baseline: Mapping[str, Any] | None = None,
    rate_threshold: float = 0.05,
    amount_threshold: float = 0.20,
) -> dict[str, Any]:
    """Create a complete monitoring report for a dataset batch."""
    snapshot = create_snapshot(frame, source)
    drift: dict[str, Any] = (
        compare_snapshots(snapshot, baseline, rate_threshold, amount_threshold)
        if baseline
        else {"status": "no_baseline", "metrics": {}, "any_exceeded": False}
    )
    return {
        "generated_at": snapshot["generated_at"],
        "kind": "descriptive_data_drift_report",
        "source": snapshot["source"],
        "rows": snapshot["rows"],
        "quality": snapshot["quality"],
        "amount": snapshot["amount"],
        "fraud": snapshot["fraud"],
        "baseline_generated_at": (baseline or {}).get("generated_at"),
        "drift": drift,
        "limitations": (
            "Descriptive threshold checks only; not a statistical drift test, "
            "concept-drift detector, or model-performance measurement."
        ),
    }
