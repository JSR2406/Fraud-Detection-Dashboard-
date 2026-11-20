"""Schema validation and quality summaries.

Built-in checks always run. Pandera is an optional additional enforcement layer:
when use_pandera=True it must be installed, and schema failures raise ValueError.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = {"TransactionID", "TransactionAmt", "isFraud"}


def pandera_available() -> tuple[bool, str]:
    try:
        import pandera
        return True, str(getattr(pandera, "__version__", "unknown"))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _pandera_module():
    try:
        from pandera import pandas as pa
    except ImportError:
        import pandera as pa
    return pa


def pandera_schema(require_target: bool = True):
    pa = _pandera_module()
    columns = {
        "TransactionID": pa.Column(pa.Int, nullable=False, unique=True),
        "TransactionAmt": pa.Column(pa.Float, nullable=False, checks=pa.Check.ge(0)),
    }
    if require_target:
        columns["isFraud"] = pa.Column(pa.Int, nullable=False, checks=pa.Check.isin([0, 1]))
    return pa.DataFrameSchema(columns, strict=False, coerce=True)


def validate_dataset(df: pd.DataFrame, require_target: bool = True, *, use_pandera: bool = False) -> None:
    required = REQUIRED if require_target else REQUIRED - {"isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    if df["TransactionID"].isna().any() or df["TransactionID"].duplicated().any():
        raise ValueError("TransactionID must be present and unique.")
    amounts = pd.to_numeric(df["TransactionAmt"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("TransactionAmt must be finite, numeric, and non-negative.")
    if require_target:
        target = set(df["isFraud"].dropna().unique())
        if df["isFraud"].isna().any() or not target.issubset({0, 1}) or target != {0, 1}:
            raise ValueError("isFraud must be non-null binary values containing both classes.")
    if use_pandera:
        installed, detail = pandera_available()
        if not installed:
            raise ImportError(f"Pandera validation requested but unavailable: {detail}")
        try:
            pandera_schema(require_target).validate(df, lazy=True)
        except Exception as exc:
            raise ValueError(f"Pandera schema validation failed: {exc}") from exc


def quality_report(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_transaction_ids": int(df["TransactionID"].duplicated().sum()) if "TransactionID" in df else None,
        "missing_cells": int(df.isna().sum().sum()),
        "missing_fraction": float(df.isna().to_numpy().mean()) if df.size else 0.0,
        "fraud_count": int(df["isFraud"].sum()) if "isFraud" in df else None,
        "fraud_rate": float(df["isFraud"].mean()) if "isFraud" in df else None,
    }
