"""
src/model_trainer.py
Train & evaluate LightGBM, Random Forest, and Logistic Regression.
Returns metrics, feature importances, and trained model artifacts.
Supports selective training via use_lgb / use_rf / use_lr flags.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def _build_lgb(X_train, y_train, X_val, y_val):
    scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = dict(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        max_depth=8, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale_pos,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    m = lgb.LGBMClassifier(**params)
    m.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def _build_rf(X_train, y_train):
    m = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    m.fit(X_train, y_train)
    return m


def _build_lr(X_train, y_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    m = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    m.fit(Xs, y_train)
    return m, scaler


def _metrics(model, X, y, scaler=None) -> dict:
    Xp   = scaler.transform(X) if scaler else X
    pred = model.predict(Xp)
    prob = model.predict_proba(Xp)[:, 1]
    cm   = confusion_matrix(y, pred)
    return {
        "accuracy":        round(accuracy_score(y, pred), 4),
        "roc_auc":         round(roc_auc_score(y, prob), 4),
        "avg_prec":        round(average_precision_score(y, prob), 4),
        "confusion_matrix": cm,
        "report":          classification_report(y, pred, output_dict=True),
    }


def train_all(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    test_size: float = 0.2,
    use_lgb: bool = True,
    use_rf:  bool = True,
    use_lr:  bool = True,
):
    """
    Train selected models and return a results dict plus X_test / y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    results = {}

    # --- LightGBM ---
    if use_lgb and HAS_LGB:
        lgb_model = _build_lgb(X_train, y_train, X_test, y_test)
        m_lgb = _metrics(lgb_model, X_test, y_test)
        m_lgb["feature_importance"] = dict(zip(feature_names, lgb_model.feature_importances_))
        results["LightGBM"] = {"model": lgb_model, "metrics": m_lgb, "scaler": None}

    # --- Random Forest ---
    if use_rf:
        rf_model = _build_rf(X_train, y_train)
        m_rf = _metrics(rf_model, X_test, y_test)
        m_rf["feature_importance"] = dict(zip(feature_names, rf_model.feature_importances_))
        results["Random Forest"] = {"model": rf_model, "metrics": m_rf, "scaler": None}

    # --- Logistic Regression ---
    if use_lr:
        lr_model, lr_scaler = _build_lr(X_train, y_train)
        m_lr = _metrics(lr_model, X_test, y_test, scaler=lr_scaler)
        m_lr["feature_importance"] = dict(zip(feature_names, np.abs(lr_model.coef_[0])))
        results["Logistic Regression"] = {"model": lr_model, "metrics": m_lr, "scaler": lr_scaler}

    if not results:
        raise ValueError("No models selected for training. Enable at least one model.")

    return results, X_test, y_test
