"""
pages/1_📊_EDA.py  — Exploratory Data Analysis
Enhanced with violin plots, scatter matrix, email domain analysis, and advanced stats.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader    import load_data
from src.visualizations import (
    transaction_amount_hist, hourly_fraud_rate,
    correlation_heatmap, class_distribution_pie,
)
from src.styles import inject_global_css, section_title, footer

st.set_page_config(page_title="EDA | Fraud Detection", page_icon="📊", layout="wide")
inject_global_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 EDA Controls")
    st.markdown("---")
    st.info("Use the tabs to navigate between different analysis views.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📊 Exploratory Data Analysis</h1>
  <p>Deep-dive into the IEEE-CIS Fraud Detection dataset — distributions, correlations,
     temporal patterns and categorical breakdowns.</p>
</div>
""", unsafe_allow_html=True)

df = load_data()

# ── Dataset summary KPIs ──────────────────────────────────────────────────────
section_title("📋 Dataset Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows",        f"{len(df):,}")
c2.metric("Columns",     f"{df.shape[1]}")
c3.metric("Missing %",   f"{df.isnull().mean().mean()*100:.1f}%")
c4.metric("Fraud Rate",  f"{df['isFraud'].mean()*100:.2f}%")
c5.metric("Fraud Count", f"{int(df['isFraud'].sum()):,}")

with st.expander("📋 Column info & data types"):
    info_df = pd.DataFrame({
        "dtype":    df.dtypes.astype(str),
        "non_null": df.notnull().sum(),
        "null_%":   (df.isnull().mean()*100).round(2),
        "unique":   df.nunique(),
    })
    st.dataframe(info_df, use_container_width=True)

st.markdown("---")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Amounts", "⏱️ Time Patterns", "🏷️ Categorical",
    "🔗 Correlations", "🔬 Advanced"
])

# ── Tab 1 : Amounts ───────────────────────────────────────────────────────────
with tab1:
    section_title("Transaction Amount Distribution")
    st.plotly_chart(transaction_amount_hist(df), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        section_title("Amount Statistics by Class")
        amt_stats = df.groupby("isFraud")["TransactionAmt"].describe().T
        amt_stats.columns = ["Legitimate", "Fraud"]
        st.dataframe(amt_stats.style.format("{:.2f}"), use_container_width=True)

    with col_b:
        section_title("Violin Plot — Amount by Class")
        fig_violin = go.Figure()
        for cls, color, name in [(0, "#6C63FF", "Legitimate"), (1, "#FF4B6E", "Fraud")]:
            sub = df[df["isFraud"] == cls]["TransactionAmt"].clip(upper=2000)
            fig_violin.add_trace(go.Violin(
                y=sub, name=name, fillcolor=color,
                line_color=color, opacity=0.7, box_visible=True,
                meanline_visible=True, points=False,
            ))
        fig_violin.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
            font=dict(color="#FAFAFA"), height=320,
            margin=dict(l=40, r=30, t=30, b=30),
            title="Amount Distribution (capped at $2000)",
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    # Box plot by product code
    if "ProductCD" in df.columns:
        section_title("Amount by Product Code")
        fig_box = px.box(
            df.assign(Class=df["isFraud"].map({0:"Legitimate",1:"Fraud"})),
            x="ProductCD", y="TransactionAmt", color="Class",
            color_discrete_map={"Legitimate":"#6C63FF","Fraud":"#FF4B6E"},
            log_y=True, title="Transaction Amount by Product Code (log scale)",
        )
        fig_box.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
            font=dict(color="#FAFAFA"), margin=dict(l=40,r=30,t=50,b=40),
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ── Tab 2 : Time Patterns ─────────────────────────────────────────────────────
with tab2:
    if "TransactionDT" not in df.columns:
        st.info("TransactionDT not available in demo data.")
    else:
        df2 = df.copy()
        df2["hour"] = (df2["TransactionDT"] // 3600) % 24
        df2["day"]  = (df2["TransactionDT"] // 86400) % 7
        day_names   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            section_title("Hourly Fraud Rate")
            st.plotly_chart(hourly_fraud_rate(df), use_container_width=True)
        with col_t2:
            section_title("Daily Fraud Rate")
            grp = df2.groupby("day")["isFraud"].mean().reset_index()
            grp["day"] = grp["day"].map(lambda x: day_names[int(x)])
            fig_day = px.bar(grp, x="day", y="isFraud",
                             color="isFraud",
                             color_continuous_scale=["#6C63FF","#FF4B6E"],
                             labels={"isFraud":"Fraud Rate"},
                             title="Daily Fraud Rate (0=Mon)")
            fig_day.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
                font=dict(color="#FAFAFA"), margin=dict(l=40,r=30,t=50,b=40),
            )
            st.plotly_chart(fig_day, use_container_width=True)

        # Heatmap: hour × day
        section_title("Fraud Heatmap: Hour × Day of Week")
        pivot = df2.groupby(["day","hour"])["isFraud"].mean().reset_index()
        pivot["day_name"] = pivot["day"].map(lambda x: day_names[int(x)])
        heat_df = pivot.pivot(index="day_name", columns="hour", values="isFraud")
        fig_heat = go.Figure(go.Heatmap(
            z=heat_df.values, x=heat_df.columns, y=heat_df.index,
            colorscale=[
                [0,   "rgba(108,99,255,0.1)"],
                [0.5, "rgba(255,193,7,0.5)"],
                [1,   "#FF4B6E"],
            ],
            hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Fraud Rate: %{z:.2%}<extra></extra>",
        ))
        fig_heat.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
            font=dict(color="#FAFAFA"),
            xaxis_title="Hour of Day", yaxis_title="Day of Week",
            margin=dict(l=60,r=30,t=30,b=40),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 3 : Categorical ───────────────────────────────────────────────────────
with tab3:
    cat_cols = [c for c in ["ProductCD","card4","card6","P_emaildomain","DeviceType"] if c in df.columns]
    col_sel, _ = st.columns([2,3])
    with col_sel:
        chosen = st.selectbox("Select categorical column", cat_cols)

    grp = df.groupby(chosen)["isFraud"].agg(["count","mean"]).reset_index()
    grp.columns = [chosen, "Count", "Fraud Rate"]
    grp = grp.sort_values("Fraud Rate", ascending=False)

    fig_cat = px.bar(grp, x=chosen, y="Fraud Rate",
                     color="Fraud Rate",
                     color_continuous_scale=["#6C63FF","#FF4B6E"],
                     title=f"Fraud Rate by {chosen}",
                     hover_data=["Count"])
    fig_cat.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"), margin=dict(l=40,r=30,t=50,b=40),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    # Stacked bar: count split
    section_title(f"Transaction Volume Split by {chosen}")
    grp2 = df.groupby([chosen,"isFraud"]).size().reset_index(name="Count")
    grp2["Class"] = grp2["isFraud"].map({0:"Legitimate",1:"Fraud"})
    fig_stack = px.bar(grp2, x=chosen, y="Count", color="Class",
                       color_discrete_map={"Legitimate":"#6C63FF","Fraud":"#FF4B6E"},
                       barmode="stack",
                       title=f"Transaction Counts by {chosen}")
    fig_stack.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"), margin=dict(l=40,r=30,t=50,b=40),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # Email domain analysis
    if chosen == "P_emaildomain" and "P_emaildomain" in df.columns:
        section_title("Top Email Domains — Fraud Count")
        email_grp = df[df["isFraud"]==1]["P_emaildomain"].fillna("unknown").value_counts().head(10)
        fig_email = go.Figure(go.Bar(
            x=email_grp.values, y=email_grp.index, orientation="h",
            marker=dict(color=email_grp.values,
                        colorscale=[[0,"#6C63FF"],[1,"#FF4B6E"]]),
            hovertemplate="%{y}: %{x} fraud cases<extra></extra>",
        ))
        fig_email.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
            font=dict(color="#FAFAFA"), height=320,
            margin=dict(l=40,r=30,t=30,b=30),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_email, use_container_width=True)

# ── Tab 4 : Correlations ──────────────────────────────────────────────────────
with tab4:
    section_title("Feature Correlation Heatmap")
    st.plotly_chart(correlation_heatmap(df), use_container_width=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["TransactionID","isFraud"]]
    top_corr = (
        df[num_cols + ["isFraud"]].corr()["isFraud"]
        .drop("isFraud").abs()
        .sort_values(ascending=False).head(10)
    )

    col_aa, col_bb = st.columns(2)
    with col_aa:
        section_title("Top 10 Features Correlated with Fraud")
        st.dataframe(
            top_corr.reset_index().rename(
                columns={"index":"Feature","isFraud":"Correlation |r|"}
            ).style.bar(subset=["Correlation |r|"], color="#6C63FF")
             .format({"Correlation |r|":"{:.4f}"}),
            use_container_width=True,
        )
    with col_bb:
        section_title("Correlation Bar Chart")
        fig_corr = go.Figure(go.Bar(
            x=top_corr.values, y=top_corr.index, orientation="h",
            marker=dict(color=top_corr.values, colorscale=[[0,"#6C63FF"],[1,"#FF4B6E"]]),
            hovertemplate="<b>%{y}</b><br>|r| = %{x:.4f}<extra></extra>",
        ))
        fig_corr.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
            font=dict(color="#FAFAFA"), height=320,
            margin=dict(l=40,r=30,t=30,b=30),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ── Tab 5 : Advanced ─────────────────────────────────────────────────────────
with tab5:
    section_title("🔬 Scatter Matrix (Top Numeric Features)")
    scatter_cols = top_corr.head(4).index.tolist() + ["isFraud"]
    scatter_cols = [c for c in scatter_cols if c in df.columns]
    fig_scatter = px.scatter_matrix(
        df.sample(min(1000, len(df)), random_state=42),
        dimensions=[c for c in scatter_cols if c != "isFraud"],
        color=df.sample(min(1000, len(df)), random_state=42)["isFraud"].astype(str),
        color_discrete_map={"0":"#6C63FF","1":"#FF4B6E"},
        labels={c:c for c in scatter_cols},
        title="Scatter Matrix — Top Correlated Features",
    )
    fig_scatter.update_traces(marker=dict(size=3, opacity=0.6))
    fig_scatter.update_layout(
        paper_bgcolor="#0E1117", font=dict(color="#FAFAFA"),
        height=550, margin=dict(l=40,r=30,t=60,b=40),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    section_title("Fraud Rate vs Transaction Amount Bucket")
    df3 = df.copy()
    df3["amt_bucket"] = pd.cut(
        df3["TransactionAmt"], bins=[0,50,100,200,500,1000,5000,99999],
        labels=["<$50","$50-100","$100-200","$200-500","$500-1K","$1K-5K",">$5K"]
    )
    bucket_grp = df3.groupby("amt_bucket", observed=True)["isFraud"].agg(["mean","count"]).reset_index()
    bucket_grp.columns = ["Bucket","Fraud Rate","Count"]
    fig_bucket = make_subplots(specs=[[{"secondary_y":True}]])
    fig_bucket.add_trace(go.Bar(
        x=bucket_grp["Bucket"], y=bucket_grp["Fraud Rate"],
        name="Fraud Rate", marker_color="#FF4B6E",
        hovertemplate="%{x}: %{y:.2%}<extra></extra>",
    ), secondary_y=False)
    fig_bucket.add_trace(go.Scatter(
        x=bucket_grp["Bucket"], y=bucket_grp["Count"],
        name="Count", mode="lines+markers",
        line=dict(color="#00D4AA", width=2),
        marker=dict(size=6),
    ), secondary_y=True)
    fig_bucket.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"), margin=dict(l=40,r=60,t=40,b=40),
        title="Fraud Rate & Volume by Amount Bucket",
        legend=dict(orientation="h", y=1.08),
    )
    fig_bucket.update_yaxes(title_text="Fraud Rate", secondary_y=False, tickformat=".1%")
    fig_bucket.update_yaxes(title_text="Transaction Count", secondary_y=True)
    st.plotly_chart(fig_bucket, use_container_width=True)

footer()
