"""
pages/2_🤖_Model_Training.py
Train models, view metrics, ROC curves, PR curves and feature importances.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd

from src.data_loader    import load_data
from src.preprocessing  import preprocess
from src.model_trainer  import train_all
from src.visualizations import (
    feature_importance_bar, confusion_matrix_heatmap,
    model_metrics_bar, roc_curve_comparison, pr_curve_comparison,
)
from src.styles import inject_global_css, section_title, footer

st.set_page_config(page_title="Model Training | Fraud Detection", page_icon="🤖", layout="wide")
inject_global_css()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Training Config")
    st.markdown("---")
    sample_size = st.slider("Sample size", 1000, 20000, 5000, step=500,
                            help="Number of rows to sample for training")
    test_size   = st.slider("Test split %", 10, 40, 20,
                            help="Percentage of data held out for evaluation") / 100

    st.markdown("---")
    st.markdown("**Models to Train**")
    use_lgb = st.checkbox("⚡ LightGBM",          value=True)
    use_rf  = st.checkbox("🌲 Random Forest",      value=True)
    use_lr  = st.checkbox("📉 Logistic Regression",value=True)

    st.markdown("---")
    run_btn = st.button("🚀 Train Models", type="primary", use_container_width=True)

    if "train_results" in st.session_state:
        st.success("✅ Model trained")
        results_ss = st.session_state["train_results"]
        for name, res in results_ss.items():
            auc = res["metrics"]["roc_auc"]
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
     <strong style="color:#FF4B6E;">Logistic Regression</strong> on the fraud dataset,
     then compare metrics, ROC curves, and feature importances side-by-side.</p>
</div>
""", unsafe_allow_html=True)

st.info("👈 Adjust parameters in the sidebar, then click **🚀 Train Models**.", icon="💡")

if run_btn or "train_results" in st.session_state:
    if run_btn:
        df = load_data()
        df = df.sample(min(sample_size, len(df)), random_state=42)

        prog_bar = st.progress(0, text="Preprocessing data…")
        with st.spinner("🔄 Preprocessing…"):
            X, y, encoders, feat_names = preprocess(df, fit=True)
            st.session_state["encoders"]   = encoders
            st.session_state["feat_names"] = feat_names
        prog_bar.progress(30, text="Training models…")

        with st.spinner("🏋️ Training models (may take ~30 s)…"):
            results, X_test, y_test = train_all(
                X, y, feat_names, test_size=test_size,
                use_lgb=use_lgb, use_rf=use_rf, use_lr=use_lr,
            )
            st.session_state["train_results"] = results
            st.session_state["X_test"]  = X_test
            st.session_state["y_test"]  = y_test
        prog_bar.progress(100, text="Done!")
        st.success("✅ Training complete! Scroll down to explore results.")
        st.balloons()

    results = st.session_state["train_results"]
    y_test  = st.session_state.get("y_test")

    # ── Model comparison ──────────────────────────────────────────────────────
    section_title("📊 Model Comparison Overview")
    st.plotly_chart(model_metrics_bar(results), use_container_width=True)

    # ── ROC & PR curves ───────────────────────────────────────────────────────
    if y_test is not None:
        col_roc, col_pr = st.columns(2)
        with col_roc:
            section_title("📈 ROC Curves")
            X_test_ss = st.session_state.get("X_test")
            if X_test_ss is not None:
                st.plotly_chart(
                    roc_curve_comparison(results, X_test_ss, y_test),
                    use_container_width=True,
                )
        with col_pr:
            section_title("🎯 Precision-Recall Curves")
            X_test_ss = st.session_state.get("X_test")
            if X_test_ss is not None:
                st.plotly_chart(
                    pr_curve_comparison(results, X_test_ss, y_test),
                    use_container_width=True,
                )

    # ── Per-model details ─────────────────────────────────────────────────────
    section_title("🔍 Detailed Per-Model Results")
    tabs = st.tabs(list(results.keys()))

    for tab, (model_name, res) in zip(tabs, results.items()):
        with tab:
            m = res["metrics"]

            # KPI row
            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.metric("Accuracy",      f"{m['accuracy']:.4f}")
            kc2.metric("ROC-AUC",       f"{m['roc_auc']:.4f}")
            kc3.metric("Avg Precision", f"{m['avg_prec']:.4f}")
            # F1 from report if available
            f1 = m.get("report", {}).get("1", {}).get("f1-score", None)
            if f1 is not None:
                kc4.metric("F1 (Fraud class)", f"{f1:.4f}")
            else:
                kc4.metric("Recall (Fraud)",   "—")

            st.markdown("<br>", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    confusion_matrix_heatmap(m["confusion_matrix"], model_name),
                    use_container_width=True,
                )
            with col_b:
                if "feature_importance" in m:
                    st.plotly_chart(
                        feature_importance_bar(
                            m["feature_importance"],
                            f"Feature Importance — {model_name}",
                        ),
                        use_container_width=True,
                    )

            with st.expander("📄 Full Classification Report"):
                report_df = pd.DataFrame(m["report"]).T
                st.dataframe(
                    report_df.style.format("{:.4f}").background_gradient(
                        cmap="RdYlGn", subset=["precision","recall","f1-score"]
                    ),
                    use_container_width=True,
                )

    # ── Best model highlight ──────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best_auc  = results[best_name]["metrics"]["roc_auc"]
    section_title("🏆 Best Model")
    st.markdown(f"""
    <div class="card" style="border-left:4px solid #00D4AA;display:flex;align-items:center;gap:2rem;">
      <div style="font-size:3rem;">🥇</div>
      <div>
        <h3 style="margin:0;color:#00D4AA;">{best_name}</h3>
        <p style="margin:0.25rem 0 0;color:#8B8FA8;">
          ROC-AUC: <strong style="color:#FAFAFA;">{best_auc:.4f}</strong> &nbsp;·&nbsp;
          This model will be used automatically on the <strong>🔮 Predict</strong> page.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

footer()
