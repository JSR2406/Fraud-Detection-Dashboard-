"""
pages/3_🔮_Predict.py
Live transaction fraud prediction page.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd

from src.data_loader   import load_data
from src.preprocessing import preprocess, NUMERIC_FEATURES, CAT_FEATURES

st.set_page_config(page_title="Predict | Fraud Detection", page_icon="🔮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.fraud-box {
    background: linear-gradient(135deg,rgba(255,75,110,.15),rgba(255,75,110,.05));
    border: 1px solid #FF4B6E; border-radius:16px; padding:1.5rem; text-align:center;
}
.safe-box {
    background: linear-gradient(135deg,rgba(108,99,255,.15),rgba(108,99,255,.05));
    border: 1px solid #6C63FF; border-radius:16px; padding:1.5rem; text-align:center;
}
.prob-label { font-size:2.5rem; font-weight:700; margin:0; }
.section-title { font-size:1.3rem; font-weight:600; color:#6C63FF;
    border-left:4px solid #6C63FF; padding-left:.75rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Live Transaction Predictor")
st.caption("Enter transaction details below to get a real-time fraud probability score.")

# ── Check if model is trained ─────────────────────────────────────────────────
if "train_results" not in st.session_state:
    st.warning("⚠️ No trained model found. Please go to **🤖 Model Training** and train first!")
    st.stop()

results   = st.session_state["train_results"]
encoders  = st.session_state.get("encoders", {})
feat_names = st.session_state.get("feat_names", [])

# Pick best model (highest ROC-AUC)
best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
best_res  = results[best_name]
model     = best_res["model"]
scaler    = best_res["scaler"]

st.info(f"🏆 Using best model: **{best_name}** (ROC-AUC: {best_res['metrics']['roc_auc']:.4f})")

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📝 Transaction Details</p>', unsafe_allow_html=True)

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        TransactionAmt = st.number_input("Transaction Amount ($)", min_value=0.01, value=120.00, step=0.01)
        ProductCD      = st.selectbox("Product Code", ["W","H","C","S","R"])
        card4          = st.selectbox("Card Type", ["visa","mastercard","discover","amex"])

    with col2:
        card6       = st.selectbox("Card Category", ["debit","credit"])
        P_emaildomain = st.selectbox("Email Domain",
                       ["gmail.com","yahoo.com","hotmail.com","anonymous.com","outlook.com","unknown"])
        DeviceType  = st.selectbox("Device Type", ["desktop","mobile","unknown"])

    with col3:
        card1 = st.number_input("Card1 (bank code)", min_value=1000, max_value=18000, value=9500)
        C1    = st.number_input("C1 (addr match count)", min_value=0, max_value=3000, value=1)
        C2    = st.number_input("C2 (card count)", min_value=0, max_value=3000, value=1)

    submitted = st.form_submit_button("🔍 Predict", type="primary", use_container_width=True)

if submitted:
    # Build a single-row dataframe that matches training schema
    row = {
        "TransactionAmt": TransactionAmt,
        "ProductCD":      ProductCD,
        "card1":          card1,
        "card2":          np.nan,
        "card4":          card4,
        "card6":          card6,
        "P_emaildomain":  P_emaildomain,
        "dist1":          np.nan,
        "C1":  C1, "C2": C2, "C6": 0, "C11": 0,
        "V95": 0, "V96": 0, "V97": 0,
        "DeviceType": DeviceType,
        "TransactionDT": 43200,   # noon default
        "isFraud": 0,             # placeholder
    }
    input_df = pd.DataFrame([row])
    X_in, _, _, _ = preprocess(input_df, fit=False, encoders=encoders)

    # Align columns
    if X_in.shape[1] != len(feat_names):
        X_in = X_in[:, :len(feat_names)]

    if scaler:
        X_in = scaler.transform(X_in)

    prob = model.predict_proba(X_in)[0][1]
    pred = int(prob >= 0.5)

    st.markdown("---")
    st.markdown('<p class="section-title">🧾 Prediction Result</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if pred == 1:
            st.markdown(f"""
            <div class="fraud-box">
              <p class="prob-label" style="color:#FF4B6E;">🚨 FRAUD</p>
              <p style="color:#FF4B6E;font-size:1.6rem;font-weight:700;">{prob*100:.1f}%</p>
              <p style="color:#8B8FA8;">Fraud Probability</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-box">
              <p class="prob-label" style="color:#6C63FF;">✅ SAFE</p>
              <p style="color:#6C63FF;font-size:1.6rem;font-weight:700;">{(1-prob)*100:.1f}%</p>
              <p style="color:#8B8FA8;">Legitimate Probability</p>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        import plotly.graph_objects as go
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Fraud Score", "font": {"color": "#FAFAFA"}},
            number={"suffix": "%", "font": {"color": "#FAFAFA", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#FAFAFA"},
                "bar":  {"color": "#FF4B6E" if pred == 1 else "#6C63FF"},
                "bgcolor": "#1A1D27",
                "steps": [
                    {"range": [0, 30],  "color": "rgba(108,99,255,0.2)"},
                    {"range": [30, 60], "color": "rgba(255,193,7,0.2)"},
                    {"range": [60, 100],"color": "rgba(255,75,110,0.2)"},
                ],
                "threshold": {"line":{"color":"#FF4B6E","width":3},"value":50},
            },
        ))
        gauge.update_layout(
            paper_bgcolor="#0E1117", font_color="#FAFAFA",
            height=280, margin=dict(l=20,r=20,t=40,b=20),
        )
        st.plotly_chart(gauge, use_container_width=True)

    with st.expander("🔎 Input summary"):
        st.json(row)
