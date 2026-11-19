"""
src/data_loader.py
Backward-compatible Streamlit-facing wrapper around ``fraud_ml.data.ingestion``.

The data mode is always explicit (``settings.data_mode``). Real-data failures are
surfaced to the caller rather than silently replaced with synthetic records.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_ml.config import settings
from fraud_ml.data.ingestion import load_dataset, make_demo_data


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset …")
def load_data(
    data_dir: str | Path | None = None,
    sample_rows: int | None = None,
    mode: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Load the configured dataset through the platform ingestion layer.

    Parameters mirror the legacy loader: ``data_dir`` may still be supplied
    positionally. When omitted, the environment-backed platform settings are
    used. There is deliberately **no** implicit synthetic fallback.
    """
    resolved_mode = str(mode or settings.data_mode).lower()
    resolved_dir = str(data_dir) if data_dir is not None else str(settings.data_dir)
    resolved_seed = settings.random_seed if seed is None else int(seed)

    try:
        frame, _source = load_dataset(
            resolved_mode,
            resolved_dir,
            seed=resolved_seed,
            sample_rows=sample_rows,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load fraud data in '{resolved_mode}' mode from "
            f"'{resolved_dir}': {exc}. No silent fallback was applied. "
            "Fix the source files/configuration, or explicitly set "
            "FRAUD_DATA_MODE=demo to request synthetic demonstration data."
        ) from exc
    return frame


def dataset_source_label(mode: str | None = None) -> str:
    """Return the human-readable source label for UI banners.

    The wording intentionally matches ``fraud_ml.data.ingestion.load_dataset``
    so banners cannot disagree with the data actually loaded.
    """
    resolved_mode = str(mode or settings.data_mode).lower()
    if resolved_mode == "real":
        return "real"
    if resolved_mode == "demo":
        return "demo — synthetic demonstration only"
    return f"unknown ({resolved_mode})"


def is_demo_source(label: str | None = None) -> bool:
    """True when the dataset label denotes synthetic / demonstration records."""
    text = str(dataset_source_label() if label is None else label).lower()
    return "synthetic" in text or "demo" in text


def _make_synthetic(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Compatibility shim for the legacy private helper.

    Explicitly generates demo records; it is never used as an automatic
    fallback by :func:`load_data`.
    """
    return make_demo_data(rows=n, seed=seed)


def get_class_distribution(df: pd.DataFrame) -> dict:
    vc = df["isFraud"].value_counts()
    return {"Legitimate": int(vc.get(0, 0)), "Fraud": int(vc.get(1, 0))}
