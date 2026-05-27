"""
src/preprocessing.py
Feature engineering & preprocessing pipeline for the fraud dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Features used for modelling
NUMERIC_FEATURES = [
    "TransactionAmt", "card1", "card2", "dist1",
    "C1", "C2", "C6", "C11", "V95", "V96", "V97",
]
CAT_FEATURES = ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]
TARGET = "isFraud"


def _safe_cols(df: pd.DataFrame, cols: list) -> list:
    """Return only columns that actually exist in the dataframe."""
    return [c for c in cols if c in df.columns]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time-based features
    if "TransactionDT" in df.columns:
        df["hour"]    = (df["TransactionDT"] // 3600) % 24
        df["day"]     = (df["TransactionDT"] // 86400) % 7
        df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # Amount features
    if "TransactionAmt" in df.columns:
        df["log_amt"]     = np.log1p(df["TransactionAmt"])
        df["amt_decimal"] = df["TransactionAmt"] % 1

    return df


def preprocess(df: pd.DataFrame, fit: bool = True, encoders: dict = None):
    """
    Encode categoricals, fill missing values, return X, y, encoders.
    If fit=False, reuse existing encoders dict for inference.
    """
    df = engineer_features(df)

    if encoders is None:
        encoders = {}

    num_cols = _safe_cols(df, NUMERIC_FEATURES + ["hour", "day", "is_night", "log_amt", "amt_decimal"])
    cat_cols = _safe_cols(df, CAT_FEATURES)

    # Encode categoricals
    for col in cat_cols:
        df[col] = df[col].fillna("unknown").astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "unknown")
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

    # Fill numeric NaNs
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    feature_cols = num_cols + cat_cols
    X = df[feature_cols].values

    y = df[TARGET].values if TARGET in df.columns else None

    return X, y, encoders, feature_cols
