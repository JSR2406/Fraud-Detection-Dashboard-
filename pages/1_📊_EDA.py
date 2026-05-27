"""
pages/1_📊_EDA.py
Exploratory Data Analysis page.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader    import load_data
from src.visualizations import (
    transaction_amount_hist, hourly_fraud_rate,
    correlation_heatmap, class_distribution_pie,
)

st.set_page_config(page_title="EDA | Fraud Detection", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-title { font-size:1.3rem; font-weight:600; color:#6C63FF;
    border-left:4px solid #6C63FF; padding-left:.75rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Exploratory Data Analysis")
st.caption("Deep-dive into the IEEE-CIS Fraud Detection dataset.")

df = load_data()

# ── Dataset summary ───────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Dataset Summary</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows",    f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]}")
c3.metric("Missing %", f"{df.isnull().mean().mean()*100:.1f}%")
c4.metric("Fraud Rate", f"{df['isFraud'].mean()*100:.2f}%")

with st.expander("📋 Column info & data types"):
    info_df = pd.DataFrame({
        "dtype":   df.dtypes.astype(str),
        "non_null": df.notnull().sum(),
        "null_%":  (df.isnull().mean()*100).round(2),
        "unique":  df.nunique(),
    })
    st.dataframe(info_df, use_container_width=True)

st.markdown("---")

# ── Distributions ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Distributions</p>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Transaction Amount", "Hourly Pattern", "Categorical Features"])

with tab1:
    st.plotly_chart(transaction_amount_hist(df), use_container_width=True)

    st.markdown("**Amount statistics by class**")
    amt_stats = df.groupby("isFraud")["TransactionAmt"].describe().T
    amt_stats.columns = ["Legitimate", "Fraud"]
    st.dataframe(amt_stats.style.format("{:.2f}"), use_container_width=True)

with tab2:
    if "TransactionDT" in df.columns:
        st.plotly_chart(hourly_fraud_rate(df), use_container_width=True)

        df2 = df.copy()
        df2["day"] = (df2["TransactionDT"] // 86400) % 7
        day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        grp = df2.groupby("day")["isFraud"].mean().reset_index()
        grp["day"] = grp["day"].map(lambda x: day_names[int(x)])
        fig = px.bar(grp, x="day", y="isFraud", title="Daily Fraud Rate",
                     labels={"isFraud": "Fraud Rate"}, color="isFraud",
                     color_continuous_scale=["#6C63FF", "#FF4B6E"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("TransactionDT not available.")

with tab3:
    cat_cols = [c for c in ["ProductCD","card4","card6","DeviceType"] if c in df.columns]
    chosen = st.selectbox("Select column", cat_cols)
    grp = df.groupby(chosen)["isFraud"].agg(["count","mean"]).reset_index()
    grp.columns = [chosen, "Count", "Fraud Rate"]
    fig = px.bar(grp, x=chosen, y="Fraud Rate", color="Fraud Rate",
                 color_continuous_scale=["#6C63FF","#FF4B6E"],
                 title=f"Fraud Rate by {chosen}")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Correlations ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Correlation Analysis</p>', unsafe_allow_html=True)
st.plotly_chart(correlation_heatmap(df), use_container_width=True)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ["TransactionID", "isFraud"]]
top_corr = df[num_cols + ["isFraud"]].corr()["isFraud"].drop("isFraud").abs().sort_values(ascending=False).head(10)
st.markdown("**Top 10 features correlated with fraud:**")
st.dataframe(top_corr.reset_index().rename(columns={"index":"Feature","isFraud":"Correlation |r|"}),
             use_container_width=True)
