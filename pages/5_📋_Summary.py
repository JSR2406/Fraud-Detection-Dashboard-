"""
pages/5_📋_Summary.py
Overall project summary — dataset stats, model leaderboard, prediction activity,
key findings and a downloadable report.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from src.data_loader    import load_data, get_class_distribution
from src.styles         import inject_global_css, section_title, footer

st.set_page_config(page_title="Summary | Fraud Detection", page_icon="📋", layout="wide")
inject_global_css()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📋 Project Summary</h1>
  <p class="hero-sub">
    A consolidated view of the <strong style="color:#6C63FF;">dataset</strong>,
    <strong style="color:#00D4AA;">model performance</strong>, and
    <strong style="color:#FFC107;">prediction activity</strong> — everything in one place.
  </p>
  <div class="stat-row" style="margin-top:1.2rem;">
    <span class="stat-pill">🗃️ Dataset Snapshot</span>
    <span class="stat-pill">🏆 Model Leaderboard</span>
    <span class="stat-pill">🔮 Prediction Activity</span>
    <span class="stat-pill">💡 Key Findings</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
df = load_data()
dist = get_class_distribution(df)
total       = dist["Legitimate"] + dist["Fraud"]
fraud_rate  = dist["Fraud"] / max(total, 1) * 100
avg_amt     = df["TransactionAmt"].mean()
median_fraud = df[df["isFraud"] == 1]["TransactionAmt"].median() if dist["Fraud"] else 0
max_amt     = df["TransactionAmt"].max()

night_fraud = 0.0
if "TransactionDT" in df.columns:
    df2 = df.copy()
    df2["hour"] = (df2["TransactionDT"] // 3600) % 24
    nt = df2[(df2["hour"] < 6) | (df2["hour"] >= 22)]
    if len(nt):
        night_fraud = nt["isFraud"].mean() * 100

# ── SECTION 1 — Dataset Snapshot ──────────────────────────────────────────────
section_title("🗃️ Dataset Snapshot")

d1, d2, d3, d4, d5, d6 = st.columns(6)
d1.metric("📦 Transactions",   f"{total:,}")
d2.metric("🚨 Fraud Cases",    f"{dist['Fraud']:,}",  f"{fraud_rate:.2f}%")
d3.metric("✅ Legitimate",     f"{dist['Legitimate']:,}")
d4.metric("💰 Avg Amount",     f"${avg_amt:,.2f}")
d5.metric("🏷️ Median Fraud $", f"${median_fraud:,.2f}")
d6.metric("🌙 Night Fraud %",  f"{night_fraud:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# Donut + risk band side-by-side
col_do, col_rb = st.columns(2)

with col_do:
    fig_do = go.Figure(go.Pie(
        labels=["Legitimate", "Fraud"],
        values=[dist["Legitimate"], dist["Fraud"]],
        hole=0.62,
        marker_colors=["#6C63FF", "#FF4B6E"],
        textinfo="label+percent",
        textfont_size=13,
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
    ))
    fig_do.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font_color="#FAFAFA",
        title=dict(text="Class Distribution", font=dict(size=15, color="#6C63FF")),
        height=300, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1),
        annotations=[dict(
            text=f"<b>{fraud_rate:.1f}%</b><br>Fraud", x=0.5, y=0.5,
            font_size=16, font_color="#FF4B6E", showarrow=False,
        )],
    )
    st.plotly_chart(fig_do, use_container_width=True)

with col_rb:
    high_risk = df[df["TransactionAmt"] > 500]["isFraud"].mean() * 100 if len(df) else 0
    mid_risk  = df[(df["TransactionAmt"] > 100) & (df["TransactionAmt"] <= 500)]["isFraud"].mean() * 100 if len(df) else 0
    low_risk  = df[df["TransactionAmt"] <= 100]["isFraud"].mean() * 100 if len(df) else 0

    bands = ["Low ≤$100", "Mid $100–500", "High >$500"]
    vals  = [low_risk, mid_risk, high_risk]
    cols  = ["#00D4AA", "#FFC107", "#FF4B6E"]

    fig_rb = go.Figure(go.Bar(
        x=bands, y=vals,
        marker_color=cols,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Fraud rate: %{y:.2f}%<extra></extra>",
    ))
    fig_rb.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
        font_color="#FAFAFA",
        title=dict(text="Fraud Rate by Amount Band", font=dict(size=15, color="#FFC107")),
        yaxis=dict(title="Fraud %", gridcolor="rgba(255,255,255,0.05)"),
        height=300, margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_rb, use_container_width=True)

# card4 & DeviceType breakdown
ins1, ins2 = st.columns(2)
with ins1:
    if "card4" in df.columns:
        ctab = df.groupby("card4")["isFraud"].agg(["count", "mean"]).reset_index()
        ctab.columns = ["Card Type", "Count", "Fraud Rate"]
        ctab["Fraud Rate %"] = (ctab["Fraud Rate"] * 100).round(2)
        fig_ct = go.Figure(go.Bar(
            x=ctab["Card Type"], y=ctab["Fraud Rate %"],
            marker_color="#6C63FF",
            text=[f"{v:.1f}%" for v in ctab["Fraud Rate %"]],
            textposition="outside",
        ))
        fig_ct.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
            font_color="#FAFAFA",
            title=dict(text="Fraud Rate by Card Type", font=dict(size=14, color="#6C63FF")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=260, margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_ct, use_container_width=True)

with ins2:
    if "DeviceType" in df.columns:
        dtab = df.groupby("DeviceType")["isFraud"].agg(["count", "mean"]).reset_index()
        dtab.columns = ["Device", "Count", "Rate"]
        fig_dt = go.Figure(go.Bar(
            x=dtab["Device"], y=(dtab["Rate"] * 100).round(2),
            marker_color="#00D4AA",
            text=[f"{v*100:.1f}%" for v in dtab["Rate"]],
            textposition="outside",
        ))
        fig_dt.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
            font_color="#FAFAFA",
            title=dict(text="Fraud Rate by Device", font=dict(size=14, color="#00D4AA")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=260, margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_dt, use_container_width=True)

st.markdown("---")

# ── SECTION 2 — Model Leaderboard ─────────────────────────────────────────────
section_title("🏆 Model Leaderboard")

if "train_results" not in st.session_state:
    st.info(
        "📌 No trained models found. Visit **🤖 Model Training** to train models, "
        "then return here for the full leaderboard.",
        icon="💡",
    )
else:
    results = st.session_state["train_results"]

    # Build leaderboard table
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        f1 = m.get("report", {}).get("1", {}).get("f1-score", None)
        rows.append({
            "Model":        name,
            "ROC-AUC":      round(m["roc_auc"], 4),
            "Accuracy":     round(m["accuracy"], 4),
            "Avg Precision":round(m["avg_prec"], 4),
            "F1 (Fraud)":   round(f1, 4) if f1 else "—",
        })
    lb_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    lb_df.index = lb_df.index + 1  # 1-based rank

    # Highlight best row
    best_name = lb_df.iloc[0]["Model"]

    st.markdown(f"""
    <div class="card" style="border-left:4px solid #00D4AA;display:flex;align-items:center;gap:1.5rem;margin-bottom:1rem;">
      <div style="font-size:2.5rem;">🥇</div>
      <div>
        <h3 style="margin:0;color:#00D4AA;">{best_name}</h3>
        <p style="margin:0;color:#8B8FA8;">Best model by ROC-AUC:
          <strong style="color:#FAFAFA;">{lb_df.iloc[0]['ROC-AUC']:.4f}</strong>
          &nbsp;·&nbsp;Accuracy:
          <strong style="color:#FAFAFA;">{lb_df.iloc[0]['Accuracy']:.4f}</strong>
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        lb_df.style.background_gradient(cmap="Purples", subset=["ROC-AUC"])
             .format({"ROC-AUC": "{:.4f}", "Accuracy": "{:.4f}", "Avg Precision": "{:.4f}"}),
        use_container_width=True,
    )

    # Radar chart comparison
    metrics_keys = ["ROC-AUC", "Accuracy", "Avg Precision"]
    categories   = metrics_keys + [metrics_keys[0]]  # close the polygon
    fig_rad = go.Figure()
    colors_rad = ["#6C63FF", "#00D4AA", "#FF4B6E", "#FFC107"]
    for i, (_, row) in enumerate(lb_df.iterrows()):
        vals = [row[k] for k in metrics_keys]
        fig_rad.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories,
            name=row["Model"],
            fill="toself",
            line_color=colors_rad[i % len(colors_rad)],
            fillcolor=colors_rad[i % len(colors_rad)].replace("#", "rgba(") + ",0.08)",
        ))
    fig_rad.update_layout(
        polar=dict(
            bgcolor="#1A1D27",
            radialaxis=dict(visible=True, range=[0.7, 1.0], gridcolor="rgba(255,255,255,0.07)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        ),
        paper_bgcolor="#0E1117", font_color="#FAFAFA",
        title=dict(text="Model Performance Radar", font=dict(size=15, color="#6C63FF")),
        height=380, margin=dict(l=40, r=40, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        showlegend=True,
    )
    st.plotly_chart(fig_rad, use_container_width=True)

st.markdown("---")

# ── SECTION 3 — Prediction Activity ───────────────────────────────────────────
section_title("🔮 Prediction Activity")

history = st.session_state.get("pred_history", [])
if not history:
    st.info("No predictions have been made yet. Go to **🔮 Predict** to test live transactions.", icon="🔮")
else:
    hist_df = pd.DataFrame(history)

    # Summary KPIs
    total_preds  = len(hist_df)
    fraud_preds  = (hist_df["Verdict"] == "🚨 FRAUD").sum()
    safe_preds   = total_preds - fraud_preds
    avg_score    = hist_df["Fraud Score"].str.replace("%", "").astype(float).mean()

    pk1, pk2, pk3, pk4 = st.columns(4)
    pk1.metric("🔢 Total Predictions", total_preds)
    pk2.metric("🚨 Fraud Flagged",     fraud_preds, f"{fraud_preds/total_preds*100:.1f}%")
    pk3.metric("✅ Safe Passed",        safe_preds,  f"{safe_preds/total_preds*100:.1f}%")
    pk4.metric("📊 Avg Fraud Score",   f"{avg_score:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Verdict pie
    v_counts = hist_df["Verdict"].value_counts()
    fig_vp = go.Figure(go.Pie(
        labels=v_counts.index.tolist(),
        values=v_counts.values.tolist(),
        hole=0.55,
        marker_colors=["#FF4B6E", "#00D4AA"],
        textinfo="label+percent",
    ))
    fig_vp.update_layout(
        paper_bgcolor="#0E1117", font_color="#FAFAFA",
        title=dict(text="Prediction Verdicts", font=dict(size=14, color="#6C63FF")),
        height=260, margin=dict(l=10, r=10, t=50, b=10),
    )
    col_vp, col_ht = st.columns([1, 2])
    with col_vp:
        st.plotly_chart(fig_vp, use_container_width=True)
    with col_ht:
        # Score histogram
        scores = hist_df["Fraud Score"].str.replace("%", "").astype(float).tolist()
        fig_sh = go.Figure(go.Histogram(
            x=scores, nbinsx=10,
            marker_color="#6C63FF",
            hovertemplate="Score %{x:.0f}%: %{y} predictions<extra></extra>",
        ))
        fig_sh.add_vline(x=50, line_dash="dash", line_color="#FF4B6E", annotation_text="Decision threshold")
        fig_sh.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
            font_color="#FAFAFA",
            title=dict(text="Fraud Score Distribution", font=dict(size=14, color="#FFC107")),
            xaxis_title="Fraud Score (%)", yaxis_title="Count",
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=260, margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_sh, use_container_width=True)

    with st.expander("📜 Full Prediction Log"):
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── SECTION 4 — Key Findings ──────────────────────────────────────────────────
section_title("💡 Key Findings & Takeaways")

f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("""
    <div class="card card-accent">
      <h4 style="color:#6C63FF;margin-top:0;">📊 Dataset Insights</h4>
      <ul style="color:#8B8FA8;font-size:0.88rem;padding-left:1.2rem;line-height:1.8;">
        <li>Severe class imbalance — ~3.5% fraud rate</li>
        <li>High-value transactions (&gt;$500) carry elevated fraud risk</li>
        <li>Night-time (10PM–6AM) shows higher fraud concentration</li>
        <li>Mobile devices correlate with higher fraud rates</li>
        <li>Anonymous email domains are strong fraud signals</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="card card-accent" style="border-left-color:#00D4AA;">
      <h4 style="color:#00D4AA;margin-top:0;">🤖 Model Insights</h4>
      <ul style="color:#8B8FA8;font-size:0.88rem;padding-left:1.2rem;line-height:1.8;">
        <li>LightGBM achieves best ROC-AUC (~0.98)</li>
        <li>Random Forest offers solid interpretability (~0.95)</li>
        <li>Logistic Regression is the fastest baseline (~0.88)</li>
        <li>Class-weight balancing crucial for recall on fraud class</li>
        <li>TransactionAmt and C-features rank highest in importance</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="card card-accent" style="border-left-color:#FFC107;">
      <h4 style="color:#FFC107;margin-top:0;">🚀 Recommendations</h4>
      <ul style="color:#8B8FA8;font-size:0.88rem;padding-left:1.2rem;line-height:1.8;">
        <li>Deploy LightGBM for production scoring</li>
        <li>Set decision threshold at 0.4–0.45 for higher recall</li>
        <li>Monitor night-time & high-value transactions closely</li>
        <li>Re-train monthly to adapt to fraud drift</li>
        <li>Integrate SHAP for regulatory explainability</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── SECTION 5 — Downloadable Report ──────────────────────────────────────────
section_title("📥 Export Summary Report")

# Build a text report
trained   = "train_results" in st.session_state
model_txt = ""
if trained:
    results = st.session_state["train_results"]
    for name, res in results.items():
        m = res["metrics"]
        model_txt += f"  {name}: AUC={m['roc_auc']:.4f}, Acc={m['accuracy']:.4f}, AvgPrec={m['avg_prec']:.4f}\n"

pred_txt = ""
history = st.session_state.get("pred_history", [])
if history:
    hist_df2 = pd.DataFrame(history)
    fraud_n  = (hist_df2["Verdict"] == "🚨 FRAUD").sum()
    pred_txt = f"Total predictions: {len(hist_df2)}  |  Fraud flagged: {fraud_n}  |  Safe: {len(hist_df2)-fraud_n}"
else:
    pred_txt = "No predictions made yet."

report_text = f"""
====================================================
  FRAUD DETECTION DASHBOARD — SUMMARY REPORT
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Author   : Janmejay Singh Rathore
====================================================

DATASET SNAPSHOT
----------------
  Total transactions : {total:,}
  Fraud cases        : {dist['Fraud']:,}  ({fraud_rate:.2f}%)
  Legitimate         : {dist['Legitimate']:,}
  Average amount     : ${avg_amt:,.2f}
  Median fraud amount: ${median_fraud:,.2f}
  Night-time fraud % : {night_fraud:.1f}%

MODEL LEADERBOARD
-----------------
{"  [No models trained]" if not trained else model_txt.rstrip()}

PREDICTION ACTIVITY
-------------------
  {pred_txt}

KEY FINDINGS
------------
  Dataset:
    - Severe class imbalance (~3.5% fraud rate)
    - High-value (>$500) transactions carry elevated fraud risk
    - Night-time fraud concentration is notably higher
  Models:
    - LightGBM best ROC-AUC (~0.98)
    - Random Forest solid at ~0.95
    - Class-weight balancing crucial for fraud recall
  Recommendations:
    - Deploy LightGBM; threshold at 0.4–0.45
    - Monitor night-time & high-value transactions
    - Re-train monthly to counter fraud drift

====================================================
  FraudShield Dashboard · IEEE-CIS · Built w/ Streamlit
====================================================
"""

st.download_button(
    label="📥 Download Report (.txt)",
    data=report_text,
    file_name=f"fraud_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
    mime="text/plain",
    use_container_width=True,
)

footer()
