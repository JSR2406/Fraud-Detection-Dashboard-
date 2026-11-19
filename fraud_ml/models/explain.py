"""Bounded SHAP explanations for supported tree-based models.

SHAP values explain *model output*, not causal effects of a transaction. If SHAP
or its native extension cannot load in the current environment, explanation
functions degrade gracefully instead of breaking training or inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExplanationResult:
    """Machine-readable explanation payload with an explicit availability flag."""

    available: bool
    reason: str | None = None
    global_importance: list[dict[str, Any]] = field(default_factory=list)
    local_explanations: list[dict[str, Any]] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "note": (
                "SHAP values describe model output contributions and are not causal effects."
                if self.available
                else self.reason
            ),
            "global_importance": self.global_importance,
            "local_explanations": self.local_explanations,
            "feature_names": self.feature_names,
        }


def explanation_support() -> tuple[bool, str]:
    """Check SHAP importability without raising (native extensions may be stale)."""
    try:
        import shap  # noqa: F401

        return True, "SHAP is available."
    except Exception as exc:  # ImportError or binary incompatibility
        return False, f"SHAP explanations unavailable ({type(exc).__name__}: {exc})."


def _source_name(transformed_name: str) -> str:
    """Turn a ColumnTransformer output name into a readable source feature."""
    name = str(transformed_name)
    for prefix in ("numeric__", "categorical__", "num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name


def _unwrap_tree(pipeline: Any) -> Any:
    """Return the terminal estimator of a sklearn Pipeline (or the object itself)."""
    model = pipeline
    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        model = pipeline.named_steps["model"]
    elif hasattr(pipeline, "steps") and pipeline.steps:
        model = pipeline.steps[-1][1]
    return model


def _matrix(pipeline: Any, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Transform engineered features into the model's numeric input space."""
    data = frame
    names: list[str] = list(frame.columns)
    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocessor"]
        transformed = preprocessor.transform(frame)
        data = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
        try:
            names = [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            names = [f"feature_{index}" for index in range(data.shape[1])]
    return np.asarray(data, dtype=float), names


def explain_transactions(
    pipeline: Any,
    background: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    max_background: int = 100,
    max_explain: int = 5,
    top_features: int = 8,
) -> ExplanationResult:
    """Compute bounded global and local explanations for tree-based pipelines."""
    available, reason = explanation_support()
    if not available:
        return ExplanationResult(available=False, reason=reason)

    try:
        import shap  # type: ignore

        model = _unwrap_tree(pipeline)
        if not (
            hasattr(model, "feature_importances_")
            or model.__class__.__name__.lower().startswith(("lgbm", "xgb"))
        ):
            return ExplanationResult(
                available=False,
                reason="SHAP explanation is only supported for tree-based models.",
            )

        X_background, feature_names = _matrix(pipeline, background)
        X_explain, _ = _matrix(pipeline, transactions)
        background_rows = X_background[: max(2, min(max_background, len(X_background)))]
        explain_rows = X_explain[: max(1, min(max_explain, len(X_explain)))]

        explainer = shap.TreeExplainer(model, data=background_rows)
        raw_values = explainer.shap_values(explain_rows)
        values = np.asarray(raw_values, dtype=float)
        if values.ndim == 3:
            # Binary classifiers may return (rows, features, classes).
            values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
        if values.shape[0] != len(explain_rows):
            values = values.T

        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_features]
        global_importance = [
            {
                "feature": _source_name(feature_names[index]),
                "mean_abs_shap": float(mean_abs[index]),
            }
            for index in order
        ]

        local_explanations: list[dict[str, Any]] = []
        for row_index in range(values.shape[0]):
            row = values[row_index]
            top = np.argsort(np.abs(row))[::-1][: min(top_features, len(row))]
            local_explanations.append(
                {
                    "row": row_index,
                    "contributors": [
                        {
                            "feature": _source_name(feature_names[index]),
                            "shap_value": float(row[index]),
                        }
                        for index in top
                    ],
                }
            )

        return ExplanationResult(
            available=True,
            reason=None,
            global_importance=global_importance,
            local_explanations=local_explanations,
            feature_names=[_source_name(name) for name in feature_names],
        )
    except Exception as exc:  # native SHAP failures must not break callers
        return ExplanationResult(
            available=False,
            reason=f"SHAP computation failed ({type(exc).__name__}: {exc}).",
        )
