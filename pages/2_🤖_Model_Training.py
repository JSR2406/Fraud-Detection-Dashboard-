"""
pages/2_🤖_Model_Training.py
Train models through fraud_ml.models.train, inspect validation/test metrics,
compare ROC/PR curves and feature importances, and persist the candidate artifact.
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st

from fraud_ml.config import settings
from fraud_ml.data.splitting import chronological_split
from fraud_ml.features.engineering import engineer_features
from fraud_ml.models.train import train_models
from src.data_loader import load_data, dataset_source_label, is_demo_source
from src.visualizations import (
    feature_importance_bar, confusion_matrix_heatmap,
    model_metrics_bar, roc_curve_comparison, pr_curve_comparison,
)
from src.styles import inject_global_css, section_title, footer

st.set_page_config(page_title="Model Training | Fraud Detection", page_icon="🤖", layout="wide")
inject_global_css()

SOURCE_LABEL = dataset_source_label()
if is_demo_source(SOURCE_LABEL):
    st.warning(
        "🚧 Training data is **synthetic demo data**. Metrics below are pipeline smoke-test "
        "results, **not validated real-world performance**. Use `FRAUD_DATA_MODE=real` for "
        "meaningful evaluation.",
        icon="⚠️",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _number(value: object, default: float = 0.0) -> float:
    """Return a finite float for optional metrics (e.g. undefined ROC-AUC)."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _classification_report(metrics: dict) -> dict:
    """Build a legacy-shaped classification report from platform metrics."""
    cm = np.asarray(metrics.get("confusion_matrix", [[0, 0], [0, 0]]), dtype=int)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    support0, support1 = tn + fn, tp + fp
    prec0, rec0 = tn / max(tn + fp, 1), tn / max(tn + fn, 1)
    f1_0 = 2 * prec0 * rec0 / max(prec0 + rec0, 1e-12)
    prec1 = _number(metrics.get("precision"))
    rec1 = _number(metrics.get("recall"))
    f1_1 = _number(metrics.get("f1"))
    total = max(tn + fp + fn + tp, 1)
    accuracy = (tn + tp) / total
    return {
        "0": {"precision": prec0, "recall": rec0, "f1-score": f1_0, "support": support0},
        "1": {"precision": prec1, "recall": rec1, "f1-score": f1_1, "support": support1},
        "accuracy": accuracy,
        "macro avg": {
            "precision": (prec0 + prec1) / 2,
            "recall": (rec0 + rec1) / 2,
            "f1-score": (f1_0 + f1_1) / 2,
            "support": support0 + support1,
        },
        "weighted avg": {
            "precision": (prec0 * support0 + prec1 * support1) / total,
            "recall": (rec0 * support0 + rec1 * support1) / total,
            "f1-score": (f1_0 * support0 + f1_1 * support1) / total,
            "support": support0 + support1,
        },
    }


def _feature_importance(pipeline: object) -> dict:
    """Extract importances from a fitted platform pipeline when available."""
    try:
        steps = pipeline.named_steps  # type: ignore[attr-defined]
        preprocessor = steps["preprocessor"]
        estimator = steps["model"]
        try:
            names = [str(n) for n in preprocessor.get_feature_names_out()]
        except Exception:
            names = []
        values = getattr(estimator, "feature_importances_", None)
        if values is None and getattr(estimator, "coef_", None) is not None:
            values = np.abs(np.asarray(estimator.coef_)).ravel()
        if values is None:
            return {}
        values = np.asarray(values).ravel()
        if not names or len(names) != len(values):
            names = [f"feature_{i}" for i in range(len(values))]
        return {name: float(value) for name, value in zip(names, values)}
    except Exception:
        return {}


class _CalibratedScorer:
    """Legacy-chart adapter: pipeline probabilities passed through the calibrator."""

    def __init__(self, pipeline: object, calibrator: object | None) -> None:
        self.pipeline = pipeline
        self.calibrator = calibrator

    def predict_proba(self, features: object) -> np.ndarray:
        probability = np.asarray(self.pipeline.predict_proba(features)[:, 1], dtype=float)  # type: ignore[attr-defined]
        if self.calibrator is not None:
            probability = np.asarray(self.calibrator.transform(probability), dtype=float)  # type: ignore[attr-defined]
        probability = np.clip(probability, 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])


def _platform_to_display(platform_results: dict) -> dict:
    """Adapt fraud_ml training results to the legacy chart/table contract."""
    display: dict = {}
    for name, item in platform_results.items():
        test = item.get("test", {})
        cm = np.asarray(test.get("confusion_matrix", [[0, 0], [0, 0]]), dtype=int)
        metrics = {
            "accuracy": _classification_report(test).get("accuracy", 0.0),
            "roc_auc": _number(test.get("roc_auc")),
            "avg_prec": _number(test.get("average_precision")),
            "confusion_matrix": cm,
            "report": _classification_report(test),
        }
        feature_importance = _feature_importance(item.get("pipeline"))
        if feature_importance:
            metrics["feature_importance"] = feature_importance
        display[name] = {
            "metrics": metrics,
            # Calibrated scorer keeps ROC/PR curves consistent with reported test metrics.
            "model": _CalibratedScorer(item.get("pipeline"), item.get("calibrator")),
            "scaler": None,
            "threshold": item.get("threshold"),
            "validation": item.get("validation", {}),
            "test": test,
        }
    return display


def _mlflow_run_info() -> str | None:
    """Return a short MLflow run summary when MLflow and a run are available."""
    try:
        import mlflow  # noqa: F401  (optional dependency)
    except ImportError:
        return None
    try:
        runs = mlflow.search_runs(max_results=1, order_by=["attributes.start_time DESC"])
        if runs is None or getattr(runs, "empty", True):
            return None
        row = runs.iloc[0]
        run_id = str(row.get("run_id", "unknown"))[:8]
        status = str(row.get("status", "unknown"))
        return f"run `{run_id}` · status `{status}`"
    except Exception:
        return None


def _read_metrics_report(artifact_dir: Path) -> dict:
    metrics_file = artifact_dir / "metrics.json"
    if metrics_file.is_file():
        try:
            return json.loads(metrics_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Training Config")
    st.markdown("---")
    sample_size = st.slider("Sample size", 1000, 20000, 5000, step=500,
                            help="Number of rows to sample for training")
    test_size = st.slider("Test split %", 10, 40, 20,
                          help="Legacy display setting; platform uses a chronological 70/15/15 split")
    st.caption("Platform split: 70% train / 15% validation / 15% test, ordered by TransactionDT.")

    st.markdown("---")
    st.markdown("**Models to Train**")
    use_lgb = st.checkbox("⚡ LightGBM", value=True)
    use_rf = st.checkbox("🌲 Random Forest", value=True)
    use_lr = st.checkbox("📉 Logistic Regression", value=True)

    st.markdown("---")
    run_btn = st.button("🚀 Train Models", type="primary", use_container_width=True)

    if st.session_state.get("platform_artifact"):
        st.success("✅ Artifact persisted")
        st.caption(str(st.session_state["platform_artifact"]))
    elif "train_results" in st.session_state:
        st.success("✅ Model trained (session)")
        for name, res in st.session_state["train_results"].items():
            auc = _number(res.get("metrics", {}).get("roc_auc"))
            st.markdown(
                f'<span class="badge badge-green">{name}: {auc:.3f}</span>',
                unsafe_allow_html=True,
            )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🤖 Model Training & Comparison</h1>
  <p>Train <strong style="color:#6C63FF;">LightGBM</strong>,
     <strong style="color:#00D4AA;">Random Forest</strong>, and
     <strong style="color:#FF4B6E;">Logistic Regression</strong> through the
     <code>fraud_ml</code> platform, then compare validation and held-out test metrics,
     ROC curves, and feature importances side-by-side.</p>
</div>
""", unsafe_allow_html=True)

st.info("👈 Adjust parameters in the sidebar, then click **🚀 Train Models**.", icon="💡")

# ── Training ──────────────────────────────────────────────────────────────────
if run_btn:
    selected = {
        "LightGBM": use_lgb,
        "Random Forest": use_rf,
        "Logistic Regression": use_lr,
    }
    if not any(selected.values()):
        st.error("Select at least one model in the sidebar.")
        st.stop()

    frame = load_data()
    sample = frame.sample(min(sample_size, len(frame)), random_state=settings.random_seed)
    artifact_dir = Path(str(settings.artifact_path)).parent
    progress = st.progress(0, text="Training platform candidates…")

    try:
        with st.spinner("🏋️ Training via fraud_ml.models.train.train_models …"):
            platform = train_models(sample, seed=settings.random_seed, artifact_dir=str(artifact_dir))
        progress.progress(70, text="Preparing comparison charts…")

        wanted = {name for name, enabled in selected.items() if enabled}
        shown_results = {k: v for k, v in platform["results"].items() if k in wanted}
        unavailable = wanted - set(platform["results"])
        if not shown_results:
            shown_results = dict(platform["results"])
            st.warning(
                "None of the selected models are available in this environment. "
                "Showing every platform candidate that trained successfully."
            )
        elif unavailable:
            st.caption(
                f"Unavailable in this environment (skipped): {', '.join(sorted(unavailable))}."
            )

        display_results = _platform_to_display(shown_results)

        # Recreate the held-out partition for the legacy ROC / PR chart helpers.
        try:
            parts = chronological_split(sample)
            X_test = engineer_features(parts.test.drop(columns=["isFraud"], errors="ignore"))
            y_test = parts.test["isFraud"].astype(int).to_numpy()
        except Exception:
            X_test, y_test = None, None

        report = _read_metrics_report(artifact_dir) or {
            "selection_metric": "validation average_precision",
            "candidate": platform.get("best_name", ""),
            "models": platform.get("report", {}),
        }

        st.session_state["train_results"] = display_results
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["platform_report"] = report
        st.session_state["platform_best"] = platform.get("best_name", "")
        st.session_state["platform_artifact"] = platform.get("artifact_path", "")
        st.session_state["platform_thresholds"] = {
            name: float(item["threshold"]) for name, item in shown_results.items()
        }
        st.session_state["platform_tracking"] = platform.get("tracking", {})
        progress.progress(100, text="Done!")
        st.success("✅ Training complete! Artifact written to the configured path.")
        st.balloons()
    except Exception as exc:
        progress.empty()
        st.error(
            "Training could not be completed: "
            f"{exc}. Check that the dataset has TransactionDT and both classes in every "
            "chronological partition, and that the sample is large enough."
        )

# ── Results (session or just-trained) ─────────────────────────────────────────
if "train_results" in st.session_state:
    results: dict = st.session_state["train_results"]
    report: dict = st.session_state.get("platform_report", {})
    best_platform = st.session_state.get("platform_best", "")
    artifact = st.session_state.get("platform_artifact", "")
    thresholds: dict = st.session_state.get("platform_thresholds", {})
    y_test = st.session_state.get("y_test")

    # ── Platform provenance / configuration ──────────────────────────────────
    section_title("⚙️ Platform Training Configuration")
    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    cfg1.metric("Selection metric", str(report.get("selection_metric", "n/a"))[:28])
    cfg2.metric("Champion", best_platform or "—")
    cfg3.metric("Decision threshold (app)", f"{settings.decision_threshold:.3f}")
    cfg4.metric("Persisted artifact", Path(artifact).name if artifact else "—")
    if artifact and Path(artifact).is_file():
        st.caption(f"✅ Trusted artifact available at `{artifact}` (loaded by Predict & Batch Detection).")
    else:
        st.caption("No persisted artifact found at the configured path; train to create one.")

    st.markdown("""
    <div class="card card-accent">
      <h4 style="color:#6C63FF;margin-top:0;">🎚️ Threshold strategy &amp; assumptions</h4>
      <p style="color:#8B8FA8;font-size:0.9rem;margin:0.25rem 0;">
        The platform selects each model's operating point on the <strong>validation</strong>
        partition only. Defaults: <code>max_f1</code> strategy,
        <code>min_precision=0.5</code>, review capacity capped at 5% of validation rows,
        <code>false_positive_cost=50</code>, <code>missed_fraud_cost=500</code>
        (BusinessAssumptions), and isotonic score calibration. The final test partition is
        scored once after these choices are fixed. Strategy, costs, calibration, the app
        decision threshold and split fractions are configurable via
        <code>fraud_ml.config.settings</code> and <code>train_models</code> arguments.
        No monetary savings are claimed — any business figures are assumption-derived estimates.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Validation vs final test ──────────────────────────────────────────────
    if report.get("models"):
        section_title("📋 Validation vs Final Test Metrics")
        rows = []
        for name, item in report["models"].items():
            val, test = item.get("validation", {}), item.get("test", {})
            rows.append({
                "Model": name,
                "Val ROC-AUC": round(_number(val.get("roc_auc")), 4),
                "Test ROC-AUC": round(_number(test.get("roc_auc")), 4),
                "Val Avg Prec": round(_number(val.get("average_precision")), 4),
                "Test Avg Prec": round(_number(test.get("average_precision")), 4),
                "Val F1": round(_number(val.get("f1")), 4),
                "Test F1": round(_number(test.get("f1")), 4),
                "Threshold": round(_number(item.get("threshold")), 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Final test metrics use the threshold chosen on validation only. "
            "Selection uses validation average precision; test is reported once, for review."
        )

    # ── MLflow (optional) ─────────────────────────────────────────────────────
    section_title("🧪 MLflow Run Info")
    tracking = st.session_state.get("platform_tracking", {})
    if tracking:
        backend = str(tracking.get("backend", "unknown"))
        st.caption(
            f"Platform tracking backend: `{backend}` · run "
            f"`{str(tracking.get('run_id', 'unknown'))[:8]}`. "
            + ("This is an MLflow run." if backend.lower() == "mlflow"
               else "This is a local JSON fallback, not an MLflow run.")
        )
    mlflow_info = _mlflow_run_info()
    if mlflow_info:
        st.info(f"Latest MLflow run: {mlflow_info}", icon="🧪")
    else:
        st.caption(
            "No queryable MLflow run is available (package not installed or no runs logged). "
            "The persisted local report above remains authoritative for this dashboard."
        )

    # ── Model comparison ──────────────────────────────────────────────────────
    section_title("📊 Model Comparison Overview")
    st.plotly_chart(model_metrics_bar(results), use_container_width=True)

    # ── ROC & PR curves ───────────────────────────────────────────────────────
    if y_test is not None:
        col_roc, col_pr = st.columns(2)
        X_test_ss = st.session_state.get("X_test")
        with col_roc:
            section_title("📈 ROC Curves")
            if X_test_ss is not None:
                st.plotly_chart(roc_curve_comparison(results, X_test_ss, y_test),
                                use_container_width=True)
        with col_pr:
            section_title("🎯 Precision-Recall Curves")
            if X_test_ss is not None:
                st.plotly_chart(pr_curve_comparison(results, X_test_ss, y_test),
                                use_container_width=True)

    # ── Per-model details ─────────────────────────────────────────────────────
    section_title("🔍 Detailed Per-Model Results")
    tabs = st.tabs(list(results.keys()))
    for tab, (model_name, res) in zip(tabs, results.items()):
        with tab:
            m = res["metrics"]
            val = res.get("validation", {})
            test = res.get("test", {})
            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.metric("Test ROC-AUC", f"{m['roc_auc']:.4f}")
            kc2.metric("Test Avg Precision", f"{m['avg_prec']:.4f}")
            kc3.metric("Val Avg Precision", f"{_number(val.get('average_precision')):.4f}")
            kc4.metric("Operating threshold",
                       f"{_number(thresholds.get(model_name, res.get('threshold'))):.4f}")

            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    confusion_matrix_heatmap(m["confusion_matrix"], model_name),
                    use_container_width=True,
                )
            with col_b:
                if m.get("feature_importance"):
                    st.plotly_chart(
                        feature_importance_bar(m["feature_importance"],
                                               f"Feature Importance — {model_name}"),
                        use_container_width=True,
                    )
                else:
                    st.caption("Feature importances are not exposed by this estimator.")

            with st.expander("📄 Full Classification Report (final test)"):
                report_df = pd.DataFrame(m["report"]).T
                # Plain numeric formatting only: Styler.background_gradient needs a
                # newer matplotlib than some deployments provide.
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
            with st.expander("📄 Raw final-test metric dictionary"):
                st.json({k: v for k, v in test.items() if k != "confusion_matrix"})

    # ── Best model highlight ──────────────────────────────────────────────────
    best_name = max(
        results,
        key=lambda k: _number(results[k]["metrics"]["roc_auc"]),
        default="",
    )
    best_auc = _number(results[best_name]["metrics"]["roc_auc"]) if best_name else 0.0
    section_title("🏆 Best Model")
    champion_note = (
        f"Persisted champion: <strong style=\"color:#FAFAFA;\">{best_platform}</strong> "
        "(selected on validation average precision). "
        if best_platform else ""
    )
    st.markdown(f"""
    <div class="card" style="border-left:4px solid #00D4AA;display:flex;align-items:center;gap:2rem;">
      <div style="font-size:3rem;">🥇</div>
      <div>
        <h3 style="margin:0;color:#00D4AA;">{best_name or '—'}</h3>
        <p style="margin:0.25rem 0 0;color:#8B8FA8;">
          Test ROC-AUC: <strong style="color:#FAFAFA;">{best_auc:.4f}</strong> &nbsp;·&nbsp;
          {champion_note}
          The persisted artifact is used automatically on the <strong>🔮 Predict</strong>
          and <strong>📦 Batch Detection</strong> pages.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

footer()
