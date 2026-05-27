"""
pages/2_🤖_Model_Training.py
Train models, view metrics and feature importances.
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
    feature_importance_bar, confusion_matrix_heatmap, model_metrics_bar
)

st.set_page_config(page_title="Model Training | Fraud Detection", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-title { font-size:1.3rem; font-weight:600; color:#6C63FF;
    border-left:4px solid #6C63FF; padding-left:.75rem; margin-bottom:1rem; }
.metric-card {
    background: linear-gradient(135deg,#1A1D27,#12151F);
    border:1px solid rgba(108,99,255,.25); border-radius:14px;
    padding:1rem 1.4rem; text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Model Training & Comparison")
st.caption("Train LightGBM, Random Forest, and Logistic Regression on the fraud dataset.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Training Config")
    sample_size = st.slider("Sample size", 1000, 20000, 5000, step=500)
    test_size   = st.slider("Test split %", 10, 40, 20) / 100
    st.markdown("---")
    run_btn = st.button("🚀 Train Models", type="primary", use_container_width=True)

st.info("👈 Adjust parameters in the sidebar, then click **Train Models**.")

if run_btn or "train_results" in st.session_state:
    if run_btn:
        df = load_data()
        df = df.sample(min(sample_size, len(df)), random_state=42)

        with st.spinner("🔄 Preprocessing data …"):
            X, y, encoders, feat_names = preprocess(df, fit=True)
            st.session_state["encoders"]    = encoders
            st.session_state["feat_names"]  = feat_names

        with st.spinner("🏋️ Training models (this may take ~30 s) …"):
            results, X_test, y_test = train_all(X, y, feat_names, test_size=test_size)
            st.session_state["train_results"] = results
            st.session_state["X_test"]  = X_test
            st.session_state["y_test"]  = y_test
        st.success("✅ Training complete!")

    results = st.session_state["train_results"]

    # ── Model comparison bar chart ────────────────────────────────────────────
    st.markdown('<p class="section-title">📊 Model Comparison</p>', unsafe_allow_html=True)
    st.plotly_chart(model_metrics_bar(results), use_container_width=True)

    # ── Per-model details ─────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🔍 Detailed Results</p>', unsafe_allow_html=True)
    tabs = st.tabs(list(results.keys()))

    for tab, (model_name, res) in zip(tabs, results.items()):
        with tab:
            m = res["metrics"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy",     f"{m['accuracy']:.4f}")
            c2.metric("ROC-AUC",      f"{m['roc_auc']:.4f}")
            c3.metric("Avg Precision",f"{m['avg_prec']:.4f}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    confusion_matrix_heatmap(m["confusion_matrix"], model_name),
                    use_container_width=True,
                )
            with col_b:
                if "feature_importance" in m:
                    st.plotly_chart(
                        feature_importance_bar(m["feature_importance"],
                                               f"Feature Importance — {model_name}"),
                        use_container_width=True,
                    )

            with st.expander("Classification Report"):
                report_df = pd.DataFrame(m["report"]).T
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
