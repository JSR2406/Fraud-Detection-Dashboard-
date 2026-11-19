"""Probability calibration for fraud scores.

Uncalibrated tree scores are useful for ranking but must not be presented as
reliable probabilities. Fit calibration on validation data (or validation-style
cross-validation) only; the final test partition stays untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


CalibrationMethod = Literal["none", "sigmoid", "isotonic"]


@dataclass(frozen=True)
class CalibrationReport:
    """Result of a calibration fit, including an audit-friendly summary."""

    method: str
    brier_before: float
    brier_after: float
    curve: dict[str, list[float]]
    fitted_estimator: Any = field(repr=False)
    cv_folds: int = 0

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe metadata (never includes the estimator itself)."""
        return {
            "method": self.method,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "cv_folds": self.cv_folds,
            "curve": self.curve,
        }


class ScoreCalibrator:
    """Post-hoc calibration fitted on validation *scores* (cheap and persistable).

    This is distinct from :func:`fit_calibration`, which cross-validates a full
    estimator. Score calibration is what the persisted artifact applies at
    inference time so probabilities are not reported as uncalibrated ranking
    scores.
    """

    def __init__(self, method: CalibrationMethod = "isotonic", seed: int = 42) -> None:
        if method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError("method must be 'none', 'sigmoid', or 'isotonic'")
        self.method = method
        self.seed = seed
        self._model: Any = None

    def fit(self, y_true: np.ndarray | list[int], probability: np.ndarray | list[float]) -> "ScoreCalibrator":
        y = np.asarray(y_true, dtype=int)
        p = np.asarray(probability, dtype=float)
        if len(y) != len(p) or len(y) == 0:
            raise ValueError("Calibration inputs must be non-empty and aligned.")
        if self.method == "none" or len(np.unique(y)) < 2:
            return self
        if self.method == "isotonic":
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._model.fit(p, y)
        else:
            clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            self._model = LogisticRegression(max_iter=1000, random_state=self.seed)
            self._model.fit(logits, y)
        return self

    def transform(self, probability: np.ndarray | list[float]) -> np.ndarray:
        p = np.asarray(probability, dtype=float)
        if self._model is None:
            return p
        if self.method == "isotonic":
            return np.asarray(self._model.predict(p), dtype=float)
        clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return np.asarray(self._model.predict_proba(logits)[:, 1], dtype=float)

    def as_dict(self) -> dict[str, Any]:
        return {"method": self.method, "seed": self.seed}


def fit_score_calibration(
    y_true: np.ndarray | list[int],
    probability: np.ndarray | list[float],
    *,
    method: CalibrationMethod = "isotonic",
    seed: int = 42,
) -> tuple[ScoreCalibrator, dict[str, Any]]:
    """Fit validation-only score calibration and report Brier score/curve."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    calibrator = ScoreCalibrator(method=method, seed=seed).fit(y, p)
    calibrated = calibrator.transform(p)
    brier_before = float(brier_score_loss(y, p))
    report = {
        "method": method if len(np.unique(y)) > 1 else f"{method}-skipped-single-class",
        "brier_before": brier_before,
        "brier_after": float(brier_score_loss(y, calibrated)),
        "curve": calibration_curve_data(y, calibrated),
    }
    return calibrator, report


def calibration_curve_data(
    y_true: np.ndarray | list[int], probability: np.ndarray | list[float], n_bins: int = 10
) -> dict[str, list[float]]:
    """Return calibration curve coordinates for plotting or JSON reporting."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid labels or probabilities for calibration curve.")
    if len(np.unique(y)) < 2:
        return {"predicted": [], "observed": []}
    bins = min(max(2, n_bins), max(2, len(y) // 2))
    observed, predicted = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return {
        "predicted": [float(value) for value in predicted],
        "observed": [float(value) for value in observed],
    }


def fit_calibration(
    estimator: Any,
    X: Any,
    y: np.ndarray | list[int],
    *,
    method: CalibrationMethod = "isotonic",
    cv: int = 5,
    seed: int = 42,
) -> CalibrationReport:
    """Fit Platt (sigmoid) or isotonic calibration on validation data.

    ``method='none'`` keeps the estimator unchanged and only reports the
    baseline Brier score/curve, so callers cannot mistake it for calibration.
    """
    if method not in {"none", "sigmoid", "isotonic"}:
        raise ValueError("method must be 'none', 'sigmoid', or 'isotonic'")
    labels = np.asarray(y, dtype=int)
    if len(labels) < 2 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Calibration requires both classes in the validation labels.")

    raw_probability = np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)
    curve = calibration_curve_data(labels, raw_probability)
    brier_before = float(brier_score_loss(labels, raw_probability))

    if method == "none":
        return CalibrationReport(method, brier_before, brier_before, curve, estimator, 0)

    class_counts = np.bincount(labels, minlength=2)
    folds = int(min(max(2, cv), int(class_counts.min())))
    fitted: Any
    if folds < 2:
        # Not enough validation rows for cross-validated calibration: keep the
        # uncalibrated estimator but record the downgrade in the report.
        return CalibrationReport(
            method=f"{method}-skipped-insufficient-validation-data",
            brier_before=brier_before,
            brier_after=brier_before,
            curve=curve,
            fitted_estimator=estimator,
            cv_folds=0,
        )

    try:
        calibrated = CalibratedClassifierCV(
            estimator=estimator, method=method, cv=folds, ensemble=True
        )
        calibrated.fit(X, labels)
    except TypeError:  # scikit-learn < 1.3 compatibility
        calibrated = CalibratedClassifierCV(
            base_estimator=estimator, method=method, cv=folds, ensemble=True
        )
        calibrated.fit(X, labels)
    except (ValueError, RuntimeError) as exc:
        # Never silently claim calibration happened if the fit failed.
        return CalibrationReport(
            method=f"{method}-skipped-{type(exc).__name__}",
            brier_before=brier_before,
            brier_after=brier_before,
            curve=curve,
            fitted_estimator=estimator,
            cv_folds=0,
        )

    calibrated_probability = np.asarray(calibrated.predict_proba(X)[:, 1], dtype=float)
    return CalibrationReport(
        method=method,
        brier_before=brier_before,
        brier_after=float(brier_score_loss(labels, calibrated_probability)),
        curve=calibration_curve_data(labels, calibrated_probability),
        fitted_estimator=calibrated,
        cv_folds=folds,
    )
