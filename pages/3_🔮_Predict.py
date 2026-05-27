"""
pages/3_🔮_Predict.py
Live transaction fraud prediction with feature contributions & prediction history.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data_loader   import load_data
from src.preprocessing import preprocess, NUMERIC_FEATURES, CAT_FEATURES
from src.styles        import inject_global_css, section_title, footer

st.set_page_config(page_title="Predict | Fraud Detection", page_icon="🔮", layout="wide")
inject_global_css()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔮 Live Transaction Predictor</h1>
  <p>Enter transaction details to get an instant <strong style="color:#FF4B6E;">fraud probability score</strong>,
     feature contribution breakdown, and a running history of all predictions.</p>
</div>
""", unsafe_allow_html=True)

# ── Guard: model must be trained ─────────────────────────────────────────────
if "train_results" not in st.session_state:
    st.warning(
        "⚠️ No trained model found. Please visit **🤖 Model Training** and train a model first!",
        icon="🚨",
    )
    st.stop()

results    = st.session_state["train_results"]
encoders   = st.session_state.get("encoders", {})
feat_names = st.session_state.get("feat_names", [])

# Pick best model (highest ROC-AUC)
best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
best_res  = results[best_name]
model     = best_res["model"]
scaler    = best_res["scaler"]

# ── Model info banner ─────────────────────────────────────────────────────────
col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("🏆 Active Model",    best_name)
col_info2.metric("📈 ROC-AUC",         f"{best_res['metrics']['roc_auc']:.4f}")
col_info3.metric("🎯 Avg Precision",   f"{best_res['metrics']['avg_prec']:.4f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar: quick presets ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎮 Scenario Presets")
    st.markdown("---")
    preset = st.selectbox("Load a preset", [
        "— None —",
        "🟢 Typical Legitimate",
        "🔴 Likely Fraud",
        "🟡 Borderline Case",
    ])
    st.markdown("---")
    st.markdown("### 📜 Prediction History")
    if "pred_history" in st.session_state and st.session_state["pred_history"]:
        st.markdown(f"**{len(st.session_state['pred_history'])} predictions made**")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["pred_history"] = []
            st.rerun()
    else:
        st.caption("No predictions yet.")

# Preset defaults
PRESETS = {
    "🟢 Typical Legitimate": dict(amt=89.99,  pcd="W", c4="visa",       c6="debit",  email="gmail.com",   device="desktop", card1=9500, C1=1,   C2=1),
    "🔴 Likely Fraud":       dict(amt=892.50, pcd="C", c4="mastercard", c6="credit", email="anonymous.com",device="mobile",  card1=3200, C1=850, C2=720),
    "🟡 Borderline Case":    dict(amt=245.00, pcd="H", c4="discover",   c6="debit",  email="yahoo.com",   device="desktop", card1=6800, C1=45,  C2=30),
}
p = PRESETS.get(preset, {})

# ── Input form ────────────────────────────────────────────────────────────────
section_title("📝 Transaction Details")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💳 Transaction Info**")
        TransactionAmt = st.number_input("Amount ($)", min_value=0.01,
                                         value=float(p.get("amt", 120.00)), step=0.01)
        ProductCD      = st.selectbox("Product Code", ["W","H","C","S","R"],
                                      index=["W","H","C","S","R"].index(p.get("pcd","W")))
        card4          = st.selectbox("Card Type",
                                      ["visa","mastercard","discover","amex"],
                                      index=["visa","mastercard","discover","amex"].index(p.get("c4","visa")))

    with col2:
        st.markdown("**🌐 Identity Signals**")
        card6         = st.selectbox("Card Category", ["debit","credit"],
                                     index=["debit","credit"].index(p.get("c6","debit")))
        P_emaildomain = st.selectbox("Purchaser Email Domain",
                                     ["gmail.com","yahoo.com","hotmail.com","anonymous.com","outlook.com","unknown"],
                                     index=["gmail.com","yahoo.com","hotmail.com","anonymous.com","outlook.com","unknown"].index(p.get("email","gmail.com")))
        DeviceType    = st.selectbox("Device Type", ["desktop","mobile","unknown"],
                                     index=["desktop","mobile","unknown"].index(p.get("device","desktop")))

    with col3:
        st.markdown("**🔢 Behaviour Counts**")
        card1 = st.number_input("Card1 (bank code)", min_value=1000, max_value=18000,
                                 value=int(p.get("card1", 9500)))
        C1    = st.number_input("C1 (address-match count)", min_value=0, max_value=3000,
                                 value=int(p.get("C1", 1)))
        C2    = st.number_input("C2 (card count)", min_value=0, max_value=3000,
                                 value=int(p.get("C2", 1)))

    submitted = st.form_submit_button("🔍 Run Prediction", type="primary", use_container_width=True)

# ── Prediction logic ──────────────────────────────────────────────────────────
if submitted:
    row = {
        "TransactionAmt": TransactionAmt,
        "ProductCD":      ProductCD,
        "card1":          card1,
        "card2":          np.nan,
        "card4":          card4,
        "card6":          card6,
        "P_emaildomain":  P_emaildomain,
        "dist1":          np.nan,
        "C1": C1, "C2": C2, "C6": 0, "C11": 0,
        "V95": 0, "V96": 0, "V97": 0,
        "DeviceType":    DeviceType,
        "TransactionDT": 43200,   # noon default
        "isFraud":       0,       # placeholder
    }
    input_df = pd.DataFrame([row])
    X_in, _, _, _ = preprocess(input_df, fit=False, encoders=encoders)

    # Align feature columns
    if X_in.shape[1] != len(feat_names):
        X_in = X_in[:, :len(feat_names)]
    if scaler:
        X_in = scaler.transform(X_in)

    prob  = model.predict_proba(X_in)[0][1]
    pred  = int(prob >= 0.5)
    score = prob * 100

    # Determine risk tier
    if score >= 70:
        risk_tier, risk_color, risk_label = "HIGH", "#FF4B6E", "🚨 HIGH RISK"
    elif score >= 40:
        risk_tier, risk_color, risk_label = "MEDIUM", "#FFC107", "⚠️ MEDIUM RISK"
    else:
        risk_tier, risk_color, risk_label = "LOW", "#00D4AA", "✅ LOW RISK"

    st.markdown("---")
    section_title("🧾 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns([1, 1.5, 1.5])

    with res_col1:
        box_class = "fraud-box" if pred == 1 else "safe-box"
        label_txt = "🚨 FRAUD" if pred == 1 else "✅ SAFE"
        label_col = "#FF4B6E" if pred == 1 else "#6C63FF"
        pct_display = f"{score:.1f}%" if pred == 1 else f"{(100-score):.1f}%"
        sub_text    = "Fraud Probability" if pred == 1 else "Legitimate Probability"
        st.markdown(f"""
        <div class="{box_class}">
          <p class="prob-label" style="color:{label_col};">{label_txt}</p>
          <p style="color:{label_col};font-size:2.2rem;font-weight:800;margin:0.25rem 0;">{pct_display}</p>
          <p style="color:#8B8FA8;margin:0;">{sub_text}</p>
          <div style="margin-top:0.75rem;">
            <span class="badge" style="background:rgba(0,0,0,0.3);border-color:{risk_color};color:{risk_color};">
              {risk_label}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Fraud Score", "font": {"color": "#FAFAFA", "size": 14}},
            number={"suffix": "%", "font": {"color": "#FAFAFA", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#FAFAFA"},
                "bar":  {"color": risk_color},
                "bgcolor": "#1A1D27",
                "steps": [
                    {"range": [0,  40], "color": "rgba(0,212,170,0.15)"},
                    {"range": [40, 70], "color": "rgba(255,193,7,0.15)"},
                    {"range": [70,100], "color": "rgba(255,75,110,0.15)"},
                ],
                "threshold": {"line": {"color": "#FF4B6E", "width": 3}, "value": 50},
            },
        ))
        gauge.update_layout(
            paper_bgcolor="#0E1117", font_color="#FAFAFA",
            height=260, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(gauge, use_container_width=True)

    with res_col3:
        # Feature contribution approximation (|feature value – population mean| × importance)
        section_title("🔬 Top Contributing Signals")
        try:
            fi = best_res["metrics"].get("feature_importance", {})
            if fi and len(feat_names):
                df_ref = load_data()
                X_ref, _, _, _ = preprocess(df_ref.head(500), fit=False, encoders=encoders)
                means = X_ref.mean(axis=0)[:len(feat_names)]
                x_vals = X_in[0][:len(feat_names)]
                contribs = {}
                for i, fname in enumerate(feat_names[:len(means)]):
                    imp = fi.get(fname, 0)
                    diff = abs(float(x_vals[i]) - float(means[i]))
                    contribs[fname] = imp * diff
                top_c = sorted(contribs.items(), key=lambda x: x[1], reverse=True)[:8]
                names_c, vals_c = zip(*top_c)
                max_v = max(vals_c) or 1
                colors_c = ["#FF4B6E" if v/max_v > 0.5 else "#FFC107" if v/max_v > 0.25 else "#6C63FF"
                            for v in vals_c]
                fig_c = go.Figure(go.Bar(
                    x=list(vals_c), y=list(names_c), orientation="h",
                    marker_color=colors_c,
                    hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>",
                ))
                fig_c.update_layout(
                    paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
                    font=dict(color="#FAFAFA"), height=260,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Contribution Score",
                )
                st.plotly_chart(fig_c, use_container_width=True)
        except Exception:
            st.caption("Feature contributions unavailable.")

    with st.expander("🔎 Full Input Summary"):
        st.json(row)

    # ── Save to history ───────────────────────────────────────────────────────
    if "pred_history" not in st.session_state:
        st.session_state["pred_history"] = []
    st.session_state["pred_history"].insert(0, {
        "Timestamp":   pd.Timestamp.now().strftime("%H:%M:%S"),
        "Amount":      f"${TransactionAmt:,.2f}",
        "Product":     ProductCD,
        "Card":        f"{card4} / {card6}",
        "Device":      DeviceType,
        "Email":       P_emaildomain,
        "Fraud Score": f"{score:.1f}%",
        "Verdict":     "🚨 FRAUD" if pred == 1 else "✅ SAFE",
        "Risk":        risk_label,
    })

# ── Prediction history table ──────────────────────────────────────────────────
if "pred_history" in st.session_state and st.session_state["pred_history"]:
    st.markdown("---")
    section_title("📜 Prediction History")
    hist_df = pd.DataFrame(st.session_state["pred_history"])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

footer()
