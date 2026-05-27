"""
src/data_loader.py
Handles loading and caching of the IEEE-CIS Fraud Detection dataset.
Uses synthetic demo data when the real CSVs are not present (for Streamlit Cloud).
"""

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Synthetic demo dataset (used when real data is absent)
# ---------------------------------------------------------------------------
def _make_synthetic(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic-looking synthetic fraud dataset."""
    rng = np.random.default_rng(seed)

    transaction_amt = rng.exponential(scale=80, size=n).round(2)
    is_fraud = (rng.random(n) < 0.035).astype(int)

    # Fraud transactions tend to be higher-value
    transaction_amt = np.where(is_fraud, transaction_amt * 3.5, transaction_amt)

    product_cd = rng.choice(["W", "H", "C", "S", "R"], size=n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    card_type  = rng.choice(["visa", "mastercard", "discover", "amex"], size=n, p=[0.55, 0.3, 0.1, 0.05])
    p_emaildomain = rng.choice(
        ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", None],
        size=n, p=[0.35, 0.25, 0.15, 0.10, 0.10, 0.05],
    )

    df = pd.DataFrame(
        {
            "TransactionID":  np.arange(2987000, 2987000 + n),
            "TransactionDT":  rng.integers(86400, 15811200, size=n),
            "TransactionAmt": transaction_amt,
            "ProductCD":      product_cd,
            "card1":          rng.integers(1000, 18000, size=n),
            "card2":          rng.choice([np.nan, *range(100, 600)], size=n),
            "card4":          card_type,
            "card6":          rng.choice(["debit", "credit"], size=n, p=[0.6, 0.4]),
            "P_emaildomain":  p_emaildomain,
            "dist1":          rng.choice([np.nan, *rng.exponential(50, 200).round(1)], size=n),
            "C1":  rng.integers(0, 3000, size=n),
            "C2":  rng.integers(0, 3000, size=n),
            "C6":  rng.integers(0, 1000, size=n),
            "C11": rng.integers(0, 1000, size=n),
            "V95":  rng.integers(0, 5, size=n),
            "V96":  rng.integers(0, 5, size=n),
            "V97":  rng.integers(0, 10, size=n),
            "DeviceType": rng.choice(["desktop", "mobile", None], size=n, p=[0.45, 0.45, 0.10]),
            "isFraud":    is_fraud,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset …")
def load_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Try to load real IEEE-CIS data from `data_dir`.
    Falls back to synthetic demo data if files are not found.
    """
    import os, zipfile

    tx_path  = os.path.join(data_dir, "train_transaction.csv.zip")
    id_path  = os.path.join(data_dir, "train_identity.csv.zip")

    if os.path.exists(tx_path):
        try:
            with zipfile.ZipFile(tx_path) as z:
                tx = pd.read_csv(z.open(z.namelist()[0]), nrows=50_000)
            if os.path.exists(id_path):
                with zipfile.ZipFile(id_path) as z:
                    identity = pd.read_csv(z.open(z.namelist()[0]))
                tx = tx.merge(identity, on="TransactionID", how="left")
            return tx
        except Exception:
            pass

    # Fallback
    return _make_synthetic()


def get_class_distribution(df: pd.DataFrame) -> dict:
    vc = df["isFraud"].value_counts()
    return {"Legitimate": int(vc.get(0, 0)), "Fraud": int(vc.get(1, 0))}
