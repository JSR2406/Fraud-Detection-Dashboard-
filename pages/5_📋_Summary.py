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
from pathlib import Path

from fraud_ml.config import settings
from src.data_loader    import load_data, get_class_distribution, dataset_source_label, is_demo_source
from src.styles         import inject_global_css, section_title, footer

st.set_page_config(page_title="Summary | Fraud Detection", page_icon="📋", layout="wide")
inject_global_css()

SOURCE_LABEL = dataset_source_label()
if is_demo_source(SOURCE_LABEL):
    st.warning(
        "🚧 This summary describes **synthetic demo data**. Figures are exploratory only and "
        "are NOT validated real-world results. Set `FRAUD_DATA_MODE=real` for meaningful data.",
        icon="⚠️",
    )


def hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    """Convert a hex color such as '#6C63FF' to an rgba CSS string."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(ch * 2 for ch in hex_color)
    if len(hex_color) != 6:
        raise ValueError(f"Unsupported hex color: {hex_color}")
    r, g, b = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]
    return f"rgba({r},{g},{b},{alpha})"


# ── Header ────────────────────────────────────────────────────────────[...]
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

# ── Load dataset ──────────────────────────────────────────────────────────[...]
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


def _number_or_none(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _persisted_leaderboard() -> list[dict]:
    """Read the platform's persisted report so the leaderboard survives refreshes."""
    metrics_file = Path(str(settings.artifact_path)).parent / "metrics.json"
    if not metrics_file.is_file():
        return []
    try:
        import json
        report = json.loads(metrics_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = []
    for name, item in report.get("models", {}).items():
        test = item.get("test", {})
        matrix = test.get("confusion_matrix") or [[0, 0], [0, 0]]
        total = sum(int(value) for row in matrix for value in row)
        accuracy = (
            (int(matrix[0][0]) + int(matrix[1][1])) / total if total else None
        )
        rows.append({
            "Model": name,
            "ROC-AUC": _number_or_none(test.get("roc_auc")),
            "Accuracy": accuracy,
            "Avg Precision": _number_or_none(test.get("average_precision")),
            "F1 (Fraud)": _number_or_none(test.get("f1")),
            "Threshold": _number_or_none(item.get("threshold")),
        })
    return rows


leaderboard_rows: list[dict] = []
if "train_results" in st.session_state:
    for name, res in st.session_state["train_results"].items():
        m = res["metrics"]
        f1 = m.get("report", {}).get("1", {}).get("f1-score")
        leaderboard_rows.append({
            "Model": name,
            "ROC-AUC": _number_or_none(m.get("roc_auc")),
            "Accuracy": _number_or_none(m.get("accuracy")),
            "Avg Precision": _number_or_none(m.get("avg_prec")),
            "F1 (Fraud)": _number_or_none(f1),
            "Threshold": _number_or_none(res.get("threshold")),
        })
else:
    leaderboard_rows = _persisted_leaderboard()

if not leaderboard_rows:
    st.info(
        "📌 No trained models found. Visit **🤖 Model Training** to train models, "
        "then return here for the full leaderboard.",
        icon="💡",
    )
else:
    lb_df = pd.DataFrame(leaderboard_rows)
    lb_df = lb_df.sort_values("Avg Precision", ascending=False, na_position="last").reset_index(drop=True)
    lb_df.index = lb_df.index + 1  # 1-based rank
    best_name = lb_df.iloc[0]["Model"]
    best_ap = _number_or_none(lb_df.iloc[0]["Avg Precision"]) or 0.0

    st.markdown(f"""
    <div class="card" style="border-left:4px solid #00D4AA;display:flex;align-items:center;gap:1.5rem;margin-bottom:1rem;">
      <div style="font-size:2.5rem;">🥇</div>
      <div>
        <h3 style="margin:0;color:#00D4AA;">{best_name}</h3>
        <p style="margin:0;color:#8B8FA8;">Selected on validation average precision.
          Final-test average precision:
          <strong style="color:#FAFAFA;">{best_ap:.4f}</strong> if a persisted report is available.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(lb_df, use_container_width=True)
    st.caption(
        "Metrics come from your own training run (session results or the persisted "
        "artifact report). Threshold and cost assumptions are configurable in fraud_ml.config."
    )

    # Radar chart comparison (only models with all three values available)
    metrics_keys = ["ROC-AUC", "Accuracy", "Avg Precision"]
    radar_df = lb_df.dropna(subset=metrics_keys)
    if len(radar_df) >= 2:
        categories = metrics_keys + [metrics_keys[0]]
        fig_rad = go.Figure()
        colors_rad = ["#6C63FF", "#00D4AA", "#FF4B6E", "#FFC107"]
        for i, (_, row) in enumerate(radar_df.iterrows()):
            vals = [float(row[k]) for k in metrics_keys]
            fig_rad.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=categories, name=row["Model"], fill="toself",
                line_color=colors_rad[i % len(colors_rad)],
                fillcolor=hex_to_rgba(colors_rad[i % len(colors_rad)], 0.08),
            ))
        fig_rad.update_layout(
            polar=dict(bgcolor="#1A1D27",
                       radialaxis=dict(visible=True, range=[0.0, 1.0], gridcolor="rgba(255,255,255,0.07)"),
                       angularaxis=dict(gridcolor="rgba(255,255,255,0.07)")),
            paper_bgcolor="#0E1117", font_color="#FAFAFA",
            title=dict(text="Model Performance Radar", font=dict(size=15, color="#6C63FF")),
            height=380, margin=dict(l=40, r=40, t=60, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15), showlegend=True,
        )
        st.plotly_chart(fig_rad, use_container_width=True)
    else:
        st.caption("Radar comparison needs at least two models with complete metric sets.")

st.markdown("---")

# ── SECTION 3 — Prediction Activity ───────────────────────────────────────────
section_title("🔮 Prediction Activity")

history = st.session_state.get("pred_history", [])
if not history:
    st.info("No predictions have been made yet. Go to **🔮 Predict** to test live transactions.", icon="🔮")
else:
    hist_df = pd.DataFrame(history)
    if "Verdict" not in hist_df.columns:
        hist_df["Verdict"] = hist_df.get("Decision", pd.Series(["—"] * len(hist_df)))
    flagged = hist_df["Verdict"].astype(str).str.contains("FRAUD|REVIEW", regex=True)
    if "Fraud Score" in hist_df.columns:
        scores = pd.to_numeric(
            hist_df["Fraud Score"].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )
    elif "fraud_probability" in hist_df.columns:
        scores = pd.to_numeric(hist_df["fraud_probability"], errors="coerce") * 100
    else:
        scores = pd.Series([np.nan] * len(hist_df))

    # Summary KPIs
    total_preds = len(hist_df)
    fraud_preds = int(flagged.sum())
    safe_preds = total_preds - fraud_preds
    avg_score = float(scores.mean()) if scores.notna().any() else 0.0

    pk1, pk2, pk3, pk4 = st.columns(4)
    pk1.metric("🔢 Total Predictions", total_preds)
    pk2.metric("🚩 Sent to Review",   fraud_preds, f"{fraud_preds/total_preds*100:.1f}%")
    pk3.metric("✅ Approved",          safe_preds,  f"{safe_preds/total_preds*100:.1f}%")
    pk4.metric("📊 Avg Fraud Score",   f"{avg_score:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Verdict pie
    v_counts = hist_df["Verdict"].astype(str).value_counts()
    pie_colors = ["#FF4B6E" if "FRAUD" in label or "REVIEW" in label else "#00D4AA"
                  for label in v_counts.index]
    fig_vp = go.Figure(go.Pie(
        labels=v_counts.index.tolist(),
        values=v_counts.values.tolist(),
        hole=0.55,
        marker_colors=pie_colors,
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
        fig_sh = go.Figure(go.Histogram(
            x=scores.dropna().tolist(), nbinsx=10,
            marker_color="#6C63FF",
            hovertemplate="Score %{x:.0f}%: %{y} predictions<extra></extra>",
        ))
        fig_sh.add_vline(x=settings.decision_threshold * 100, line_dash="dash",
                         line_color="#FF4B6E", annotation_text="Decision threshold")
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
    st.markdown(f"""
    <div class="card card-accent">
      <h4 style="color:#6C63FF;margin-top:0;">📊 Dataset Snapshot</h4>
      <ul style="color:#8B8FA8;font-size:0.88rem;padding-left:1.2rem;line-height:1.8;">
        <li>Loaded source: <strong>{SOURCE_LABEL}</strong></li>
        <li>Observed fraud rate in this dataset: <strong>{fraud_rate:.2f}%</strong></li>
        <li>Night-time fraud rate (relative hour): <strong>{night_fraud:.1f}%</strong></li>
        <li>Amount, product and relative-time breakdowns are on the 📊 EDA page.</li>
        <li>Demo records are illustrative and are not real-world evidence.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="card card-accent" style="border-left-color:#00D4AA;">
      <h4 style="color:#00D4AA;margin-top:0;">🤖 Model Insights</h4>
      <ul style="color:#8B8FA8;font-size:0.88rem;padding-left:1.2rem;line-height:1.8;">
        <li>No fixed winner is asserted: the champion is selected on validation average precision.</li>
        <li>Final test metrics are reported only after that selection.</li>
        <li>Operating thresholds come from the validation partition via fraud_ml.models.threshold.</li>
        <li>Artifact loading is restricted to trusted, operator-provisioned files.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="card card-accent" style="border-left-color:#FFC107;">
      <h4 style="color:#FFC107;margin-top:0;">🚀 Recommendations</h4>
      <ul style="color:#8B8FA8;font-size:0.88rem;padding-left:1.2rem;line-height:1.8;">
        <li>Configure threshold, split fractions and cost assumptions in fraud_ml.config.</li>
        <li>Validate on licensed real data before any operational decision.</li>
        <li>Monitor drift and re-train on a schedule you define.</li>
        <li>Treat any optional model explanations as non-causal signals only.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── SECTION 5 — Downloadable Report ──────────────────────────────────────────
section_title("📥 Export Summary Report")

# Build a text report
trained   = bool(leaderboard_rows)
model_txt = ""
if trained:
    for row in leaderboard_rows:
        auc = _number_or_none(row.get("ROC-AUC"))
        ap = _number_or_none(row.get("Avg Precision"))
        model_txt += (f"  {row['Model']}: AUC={auc if auc is not None else 'n/a'}, "
                      f"AvgPrec={ap if ap is not None else 'n/a'}\n")

pred_txt = ""
history = st.session_state.get("pred_history", [])
if history:
    hist_df2 = pd.DataFrame(history)
    if "Verdict" in hist_df2.columns:
        flagged_n = int(hist_df2["Verdict"].astype(str).str.contains("FRAUD|REVIEW", regex=True).sum())
    else:
        flagged_n = 0
    pred_txt = (f"Total predictions: {len(hist_df2)}  |  Sent to review: {flagged_n}  |  "
                f"Approved: {len(hist_df2)-flagged_n}")
else:
    pred_txt = "No predictions made yet."

report_text = f"""
====================================================
  FRAUD DETECTION DASHBOARD — SUMMARY REPORT
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Author   : Janmejay Singh Rathore
====================================================

DATASET SOURCE: {SOURCE_LABEL}
(If this says demo/synthetic, no figure below is a validated real-world result.)

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

KEY FINDINGS / CAVEATS
----------------------
  - All dataset figures above describe the loaded source only.
  - Champion is selected on validation average precision; test metrics follow selection.
  - Decision threshold, split fractions and cost assumptions are configurable via fraud_ml.config.
  - Explainability, when available, is non-causal and is not evidence of fraud.
  - Validate on licensed real data before operational use.

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
