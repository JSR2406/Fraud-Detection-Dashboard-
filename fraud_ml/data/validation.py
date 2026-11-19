"""Schema validation and quality summaries."""
import numpy as np
import pandas as pd
REQUIRED = {"TransactionID", "TransactionAmt", "isFraud"}
def validate_dataset(df: pd.DataFrame, require_target: bool = True) -> None:
    required = REQUIRED if require_target else REQUIRED - {"isFraud"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    if df["TransactionID"].isna().any() or df["TransactionID"].duplicated().any(): raise ValueError("TransactionID must be present and unique.")
    amounts = pd.to_numeric(df["TransactionAmt"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any(): raise ValueError("TransactionAmt must be finite, numeric, and non-negative.")
    if require_target:
        target = set(df["isFraud"].dropna().unique())
        if df["isFraud"].isna().any() or not target.issubset({0, 1}) or target != {0, 1}: raise ValueError("isFraud must be non-null binary values containing both classes.")
def quality_report(df: pd.DataFrame) -> dict:
    return {"rows": int(len(df)), "columns": int(df.shape[1]), "duplicate_transaction_ids": int(df["TransactionID"].duplicated().sum()) if "TransactionID" in df else None, "missing_cells": int(df.isna().sum().sum()), "missing_fraction": float(df.isna().to_numpy().mean()) if df.size else 0.0, "fraud_count": int(df["isFraud"].sum()) if "isFraud" in df else None, "fraud_rate": float(df["isFraud"].mean()) if "isFraud" in df else None}
