"""Reproducible LightGBM hyperparameter optimization.

Optuna is used when installed. A bounded random-search fallback keeps the
pipeline functional in minimal environments; both backends optimize validation
PR-AUC only and never touch the final test partition.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_ml.features.pipeline import build_preprocessor


@dataclass(frozen=True)
class TuningResult:
    """Trial history and the best validation-scoring parameters."""

    best_params: dict[str, Any]
    best_score: float
    backend: str
    trials: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "trials": self.trials,
        }

    def persist(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return target


def _parameter_space(trial: Any) -> dict[str, Any]:
    """Shared search space for Optuna trials and the fallback sampler."""
    suggest = getattr(trial, "suggest_float", None)
    if suggest is not None:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
    rng = trial  # fallback: a numpy Generator
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.15)))),
        "num_leaves": int(rng.integers(15, 64)),
        "min_child_samples": int(rng.integers(10, 81)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_lambda": float(rng.uniform(0.0, 5.0)),
    }


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray | list[int],
    X_validation: pd.DataFrame,
    y_validation: np.ndarray | list[int],
    *,
    n_trials: int = 20,
    seed: int = 42,
    timeout: int | None = None,
    early_stopping_rounds: int = 30,
    log_path: str | Path | None = None,
    study_name: str = "fraud-lightgbm",
    study_storage: str | None = None,
) -> TuningResult:
    """Optimize LightGBM on the validation partition only."""
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("LightGBM is required for hyperparameter tuning.") from exc

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if len(X_train) != len(y_train) or len(X_validation) != len(y_validation):
        raise ValueError("Feature/label lengths do not match.")

    preprocessor = build_preprocessor()
    Xt = preprocessor.fit_transform(X_train, y_train)
    Xv = preprocessor.transform(X_validation)
    xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    xv = Xv.toarray() if hasattr(Xv, "toarray") else np.asarray(Xv)
    ytr = np.asarray(y_train, dtype=int)
    yva = np.asarray(y_validation, dtype=int)

    positive = int((ytr == 1).sum())
    base = {
        "n_estimators": 400,
        "objective": "binary",
        "scale_pos_weight": float(max((ytr == 0).sum(), 1) / max(positive, 1)),
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": -1,
    }

    def evaluate(params: dict[str, Any]) -> float:
        model = lgb.LGBMClassifier(**base, **params)
        model.fit(
            xt,
            ytr,
            eval_set=[(xv, yva)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        probability = model.predict_proba(xv)[:, 1]
        return float(average_precision_score(yva, probability))

    trials: list[dict[str, Any]] = []
    try:
        import optuna  # type: ignore

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_kwargs: dict[str, Any] = {
            "study_name": study_name,
            "direction": "maximize",
            "sampler": optuna.samplers.TPESampler(seed=seed),
        }
        if study_storage:
            # SQLite storage keeps studies across runs and makes trials auditable.
            study_kwargs["storage"] = study_storage
            study_kwargs["load_if_exists"] = True
        study = optuna.create_study(**study_kwargs)
        study.optimize(
            lambda trial: evaluate(_parameter_space(trial)),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )
        best_params = dict(study.best_params)
        best_score = float(study.best_value)
        backend = "optuna"
        trials = [
            {"number": t.number, "value": float(t.value), "params": dict(t.params)}
            for t in study.trials
            if t.value is not None
        ]
    except ImportError:
        rng = np.random.default_rng(seed)
        for index in range(n_trials):
            params = _parameter_space(rng)
            score = evaluate(params)
            trials.append({"number": index, "value": score, "params": params})
        best = max(trials, key=lambda item: item["value"])
        best_params = dict(best["params"])
        best_score = float(best["value"])
        backend = "numpy-random-search (install optuna for TPE studies)"

    result = TuningResult(best_params, best_score, backend, trials)
    if log_path is not None:
        result.persist(log_path)
    return result
