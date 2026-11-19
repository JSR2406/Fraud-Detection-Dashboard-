"""Dataset-quality and distribution-drift monitoring.

IEEE-CIS TransactionDT is relative elapsed time, so calendar-based
monitoring windows are not meaningful. All summaries are explicit and local.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_ml.data.validation import quality_report


def build_reference_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize a trusted reference partition for later drift comparisons."""
    profile: dict[str, Any] = {
        "rows": int(len(df)),
        "quality": quality_report(df),
        "numeric": {},
        "categorical": {},
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    for column in df.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        profile["numeric"][str(column)] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "missing_fraction": float(df[column].isna().mean()),
        }
    for column in df.select_dtypes(exclude=[np.number]).columns:
        frequencies = df[column].astype(str).value_counts(normalize=True).head(20)
        profile["categorical"][str(column)] = {
            "top_values": {str(key): float(value) for key, value in frequencies.items()},
            "missing_fraction": float(df[column].isna().mean()),
        }
    return profile


def compare_to_reference(
    df: pd.DataFrame, profile: dict[str, Any], *, mean_shift_alert: float = 2.0
) -> dict[str, Any]:
    """Compare a new batch against a reference profile and flag large shifts."""
    quality = quality_report(df)
    alerts: list[str] = []
    numeric_drift: dict[str, Any] = {}
    for column, reference in profile.get("numeric", {}).items():
        if column not in df.columns:
            alerts.append(f"Missing monitored numeric column: {column}")
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            alerts.append(f"No finite values for monitored column: {column}")
            continue
        mean = float(values.mean())
        std = float(values.std())
        reference_std = float(reference.get("std", 0.0))
        shift = abs(mean - float(reference.get("mean", 0.0))) / max(reference_std, std, 1e-9)
        changed = bool(shift > mean_shift_alert)
        numeric_drift[column] = {
            "reference_mean": reference.get("mean"),
            "observed_mean": mean,
            "mean_shift_in_std_units": float(shift),
            "alert": changed,
        }
        if changed:
            alerts.append(f"{column} mean shifted by {shift:.2f} std units")

    reference_fraud_rate = profile.get("quality", {}).get("fraud_rate")
    observed_fraud_rate = quality.get("fraud_rate")
    if (
        reference_fraud_rate is not None
        and observed_fraud_rate is not None
        and float(reference_fraud_rate) > 0
    ):
        rate_ratio = float(observed_fraud_rate) / float(reference_fraud_rate)
        if rate_ratio > 2 or rate_ratio < 0.5:
            alerts.append(f"Fraud rate ratio changed to {rate_ratio:.2f}× the reference")

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "quality": quality,
        "numeric_drift": numeric_drift,
        "alerts": alerts,
        "status": "alert" if alerts else "ok",
        "note": "TransactionDT is relative time; no calendar-window inference is made.",
    }


def write_monitoring_report(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return target
