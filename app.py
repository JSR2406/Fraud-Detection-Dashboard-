"""
app.py — Fraud Detection Dashboard
Author : Janmejay Singh Rathore
Dataset: IEEE-CIS Fraud Detection (Kaggle)
Model  : LightGBM (primary), Random Forest, Logistic Regression
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data_loader    import load_data, get_class_distribution
from src.visualizations import (
    class_distribution_pie,
    transaction_amount_hist,
    hourly_fraud_rate,
    correlation_heatmap,
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1A1D27 0%, #12151F 100%);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0E1117 0%, #1A1D27 50%, #12151F 100%);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(108,99,255,0.15);
}
.hero h1 { font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(90deg, #6C63FF, #00D4AA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
.hero p  { color: #8B8FA8; font-size: 1.05rem; margin-top: 0.5rem; }

/* Section headings */
.section-title {
    font-size: 1.3rem; font-weight: 600; color: #6C63FF;
    border-left: 4px solid #6C63FF; padding-left: 0.75rem;
    margin-bottom: 1rem;
}

/* Stat badge */
.badge {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.4);
    border-radius: 8px;
    padding: 0.25rem 0.75rem;
    font-size: 0.85rem; color: #6C63FF;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ Fraud Detection")
    st.markdown("---")
    st.markdown("""
    **Navigation**
    - 🏠 **Home** ← You are here
    - 📊 **EDA** — Data exploration
    - 🤖 **Model Training** — Train & compare
    - 🔮 **Predict** — Live transaction check
    - ℹ️ **About** — Project details
    """)
    st.markdown("---")
    st.markdown(
        '<span class="badge">LightGBM</span>'
        '<span class="badge">IEEE-CIS</span>',
        unsafe_allow_html=True,
    )
    st.caption("Janmejay Singh Rathore · 2024")

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛡️ Fraud Detection Dashboard</h1>
  <p>Real-time transaction fraud analysis powered by <strong>LightGBM</strong> on the
     IEEE-CIS dataset. Explore patterns, compare models, and predict live transactions.</p>
</div>
""", unsafe_allow_html=True)

# ── KPI metrics ───────────────────────────────────────────────────────────────
dist = get_class_distribution(df)
total  = dist["Legitimate"] + dist["Fraud"]
fraud_rate = dist["Fraud"] / max(total, 1) * 100
avg_amt = df["TransactionAmt"].mean()
median_fraud_amt = df[df["isFraud"] == 1]["TransactionAmt"].median() if dist["Fraud"] else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Total Transactions", f"{total:,}")
col2.metric("🚨 Fraud Cases",        f"{dist['Fraud']:,}", f"{fraud_rate:.2f}% rate")
col3.metric("✅ Legitimate",          f"{dist['Legitimate']:,}")
col4.metric("💰 Avg Transaction",    f"${avg_amt:,.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row 1 ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📈 Overview Analytics</p>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(class_distribution_pie(dist), use_container_width=True)
with c2:
    st.plotly_chart(transaction_amount_hist(df),  use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)
with c3:
    if "TransactionDT" in df.columns:
        st.plotly_chart(hourly_fraud_rate(df), use_container_width=True)
    else:
        st.info("TransactionDT column not available for hourly chart.")
with c4:
    st.plotly_chart(correlation_heatmap(df), use_container_width=True)

# ── Raw data preview ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-title">🗂️ Dataset Preview</p>', unsafe_allow_html=True)
with st.expander("Show sample rows"):
    n_rows = st.slider("Rows to display", 5, 100, 20)
    st.dataframe(df.head(n_rows), use_container_width=True)

st.markdown("""
---
<center style="color:#555;font-size:0.8rem;">
Built with ❤️ by Janmejay Singh Rathore · IEEE-CIS Fraud Detection · Streamlit
</center>
""", unsafe_allow_html=True)