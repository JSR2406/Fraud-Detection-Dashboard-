"""
app.py — Fraud Detection Dashboard  (Home)
Author : Janmejay Singh Rathore
Dataset: IEEE-CIS Fraud Detection (Kaggle)
Model  : LightGBM (primary), Random Forest, Logistic Regression
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from src.data_loader    import load_data, get_class_distribution
from src.visualizations import (
    class_distribution_pie,
    transaction_amount_hist,
    hourly_fraud_rate,
    correlation_heatmap,
)
from src.styles import inject_global_css, section_title, footer

# ── Inject CSS ───────────────────────────────────────────────────────────────
inject_global_css()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">🛡️</div>
        <h3 style="margin:0.25rem 0 0;letter-spacing:-0.02em;">FraudShield</h3>
        <p style="color:#8B8FA8;font-size:0.8rem;margin:0;">IEEE-CIS · LightGBM</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**📌 Navigation**")
    st.markdown("""
    <div style="display:flex;flex-direction:column;gap:4px;padding:0.25rem 0;">
      <div style="background:rgba(108,99,255,0.2);border-radius:8px;padding:0.4rem 0.75rem;border-left:3px solid #6C63FF;font-size:0.9rem;">🏠 <b>Home</b> ← you are here</div>
      <div style="padding:0.4rem 0.75rem;font-size:0.9rem;color:#8B8FA8;">📊 EDA — Data exploration</div>
      <div style="padding:0.4rem 0.75rem;font-size:0.9rem;color:#8B8FA8;">🤖 Model Training — Train &amp; compare</div>
      <div style="padding:0.4rem 0.75rem;font-size:0.9rem;color:#8B8FA8;">🔮 Predict — Live transaction check</div>
      <div style="padding:0.4rem 0.75rem;font-size:0.9rem;color:#8B8FA8;">ℹ️ About — Project details</div>
      <div style="padding:0.4rem 0.75rem;font-size:0.9rem;color:#8B8FA8;">📋 Summary — Overall summarisation</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model status
    is_trained = "train_results" in st.session_state
    status_html = (
        '<span class="badge badge-green">✅ Model Trained</span>'
        if is_trained else
        '<span class="badge badge-red">⏳ Not Trained Yet</span>'
    )
    st.markdown(f"**Model Status**<br>{status_html}", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<span class="badge">⚡ LightGBM</span>'
        '<span class="badge">🌲 RF</span>'
        '<span class="badge">📉 LR</span>',
        unsafe_allow_html=True,
    )
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    st.caption(f"Janmejay Singh Rathore · {now}")

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛡️ Fraud Detection Dashboard</h1>
  <p class="hero-sub">
    Real-time financial transaction analysis powered by <strong style="color:#6C63FF;">LightGBM</strong> on
    the <strong style="color:#00D4AA;">IEEE-CIS</strong> dataset. Explore patterns, compare models, and
    predict live transactions with explainability.
  </p>
  <div class="stat-row" style="margin-top:1.2rem;">
    <span class="stat-pill">🏦 IEEE-CIS Dataset</span>
    <span class="stat-pill">📊 3-Model Comparison</span>
    <span class="stat-pill">⚡ Real-time Prediction</span>
    <span class="stat-pill">🔍 Interactive Analytics</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI metrics ───────────────────────────────────────────────────────────────
dist = get_class_distribution(df)
total        = dist["Legitimate"] + dist["Fraud"]
fraud_rate   = dist["Fraud"] / max(total, 1) * 100
avg_amt      = df["TransactionAmt"].mean()
median_fraud = df[df["isFraud"] == 1]["TransactionAmt"].median() if dist["Fraud"] else 0
night_fraud  = 0
if "TransactionDT" in df.columns:
    df2 = df.copy()
    df2["hour"] = (df2["TransactionDT"] // 3600) % 24
    night_tx  = df2[(df2["hour"] < 6) | (df2["hour"] >= 22)]
    if len(night_tx):
        night_fraud = night_tx["isFraud"].mean() * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📦 Total Transactions", f"{total:,}")
col2.metric("🚨 Fraud Cases",        f"{dist['Fraud']:,}", f"{fraud_rate:.2f}% rate")
col3.metric("✅ Legitimate",         f"{dist['Legitimate']:,}")
col4.metric("💰 Avg Transaction",    f"${avg_amt:,.2f}")
col5.metric("🌙 Night Fraud Rate",   f"{night_fraud:.1f}%", help="Fraud % between 10PM–6AM")

st.markdown("<br>", unsafe_allow_html=True)

# ── Overview Analytics ────────────────────────────────────────────────────────
section_title("📈 Overview Analytics")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(class_distribution_pie(dist), use_container_width=True)
with c2:
    st.plotly_chart(transaction_amount_hist(df), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    if "TransactionDT" in df.columns:
        st.plotly_chart(hourly_fraud_rate(df), use_container_width=True)
    else:
        st.info("TransactionDT column not available for hourly chart.")
with c4:
    st.plotly_chart(correlation_heatmap(df), use_container_width=True)

# ── Quick insights banner ─────────────────────────────────────────────────────
section_title("💡 Quick Insights")
i1, i2, i3 = st.columns(3)
with i1:
    st.markdown("""
    <div class="card card-accent">
      <h4 style="color:#6C63FF;margin-top:0;">💳 Fraud by Card Type</h4>
    """, unsafe_allow_html=True)
    if "card4" in df.columns:
        ctab = df.groupby("card4")["isFraud"].mean().sort_values(ascending=False)
        for card, rate in ctab.items():
            bar_w = int(rate * 1000)
            color = "#FF4B6E" if rate > 0.05 else "#6C63FF"
            st.markdown(f"""
            <div style="margin:4px 0;">
              <div style="display:flex;justify-content:space-between;font-size:0.85rem;">
                <span>{card}</span><span style="color:{color};font-weight:600;">{rate*100:.1f}%</span>
              </div>
              <div style="height:4px;background:#1A1D27;border-radius:2px;margin-top:2px;">
                <div style="height:4px;width:{bar_w}%;background:{color};border-radius:2px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with i2:
    st.markdown("""
    <div class="card card-accent">
      <h4 style="color:#00D4AA;margin-top:0;">📱 Device Breakdown</h4>
    """, unsafe_allow_html=True)
    if "DeviceType" in df.columns:
        dtab = df.groupby("DeviceType")["isFraud"].agg(["count","mean"]).reset_index()
        for _, row in dtab.iterrows():
            emoji = "🖥️" if row["DeviceType"] == "desktop" else "📱" if row["DeviceType"] == "mobile" else "❓"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.4rem 0;
                        border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.88rem;">
              <span>{emoji} {row['DeviceType'].title()}</span>
              <span style="color:#8B8FA8;">{int(row['count']):,} txns &nbsp;|&nbsp;
                <span style="color:#FF4B6E;font-weight:600;">{row['mean']*100:.1f}%</span> fraud</span>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with i3:
    high_risk_pct = df[df["TransactionAmt"] > 500]["isFraud"].mean() * 100 if len(df) else 0
    low_risk_pct  = df[df["TransactionAmt"] <= 100]["isFraud"].mean() * 100 if len(df) else 0
    st.markdown(f"""
    <div class="card card-accent">
      <h4 style="color:#FFC107;margin-top:0;">💰 Amount Risk Bands</h4>
      <div style="margin:0.5rem 0;">
        <div style="display:flex;justify-content:space-between;font-size:0.88rem;padding:0.35rem 0;">
          <span>💸 High-value (&gt;$500)</span>
          <span style="color:#FF4B6E;font-weight:700;">{high_risk_pct:.1f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.88rem;padding:0.35rem 0;">
          <span>🪙 Low-value (≤$100)</span>
          <span style="color:#00D4AA;font-weight:700;">{low_risk_pct:.1f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.88rem;padding:0.35rem 0;">
          <span>📊 Overall fraud rate</span>
          <span style="color:#6C63FF;font-weight:700;">{fraud_rate:.2f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.88rem;padding:0.35rem 0;">
          <span>🚨 Median fraud amount</span>
          <span style="color:#FFC107;font-weight:700;">${median_fraud:,.2f}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Dataset preview ───────────────────────────────────────────────────────────
st.markdown("---")
section_title("🗂️ Dataset Preview")
with st.expander("Show sample rows"):
    c_a, c_b = st.columns([3,1])
    with c_a:
        n_rows = st.slider("Rows to display", 5, 100, 20)
    with c_b:
        show_fraud_only = st.checkbox("Fraud only")
    display_df = df[df["isFraud"]==1] if show_fraud_only else df
    st.dataframe(display_df.head(n_rows), use_container_width=True)

footer()