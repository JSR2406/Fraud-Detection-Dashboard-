"""
pages/3_🔮_Predict.py
Live transaction scoring via the persisted fraud_ml predictor (or an optional
FastAPI endpoint), with non-sensitive history and a guarded explanation panel.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud_ml.config import settings
from fraud_ml.features.engineering import RAW_COLUMNS
from fraud_ml.inference.predictor import FraudPredictor
from src.data_loader import dataset_source_label, is_demo_source
from src.styles import inject_global_css, section_title, footer

st.set_page_config(page_title="Predict | Fraud Detection", page_icon="🔮", layout="wide")
inject_global_css()

SOURCE_LABEL = dataset_source_label()
if is_demo_source(SOURCE_LABEL):
    st.warning(
        "🚧 Loaded data is **synthetic demo data**. Prediction scores are illustrative and "
        "are not validated real-world risk estimates.",
        icon="⚠️",
    )


# ---------------------------------------------------------------------------
# Cached model / API helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading persisted model …")
def load_predictor(path: str) -> tuple[FraudPredictor | None, str | None]:
    """Load a trusted operator-provided artifact, never an arbitrary upload."""
    artifact = Path(path)
    if not artifact.is_file():
        return None, "No persisted model artifact found. Train on the 🤖 Model Training page."
    try:
        return FraudPredictor.from_trusted_file(artifact), None
    except Exception as exc:
        return None, f"Could not load the trusted artifact: {type(exc).__name__}."


def _post_json(url: str, payload: dict, headers: dict) -> Any:
    try:
        import requests
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except ImportError:
        pass
    try:
        import httpx
        response = httpx.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except ImportError:
        return None


def _score_via_api(row: dict, api_url: str) -> dict | None:
    """Best-effort call to a FastAPI scoring endpoint; never expose the payload in logs."""
    base = api_url.strip().rstrip("/")
    if not base:
        return None
    urls = [f"{base}/api/v1/predict"] if not base.endswith("/predict") else []
    urls += [base if base.endswith("/predict") else f"{base}/predict"]
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["X-API-Key"] = settings.api_key

    # The API's single-score endpoint accepts a bare transaction object; batch
    # envelopes are also tried to stay compatible with other deployments.
    payloads = [row, {"transactions": [row]}]
    for url in dict.fromkeys(urls):
        for payload in payloads:
            try:
                data = _post_json(url, payload, headers)
            except Exception:
                continue
            if data is None:
                continue
            predictions = data
            if isinstance(data, dict):
                predictions = (
                    data.get("predictions") or data.get("results") or data.get("data") or data
                )
            if isinstance(predictions, dict):
                predictions = [predictions]
            if isinstance(predictions, list) and predictions and isinstance(predictions[0], dict):
                first = dict(predictions[0])
                required = {"fraud_probability", "risk_level", "decision", "model_version"}
                if required.issubset(first):
                    first.setdefault("transaction_id", row["transaction_id"])
                    return first
    return None


def _render_explanation(predictor: FraudPredictor | None, row: dict, result: dict) -> None:
    """Show a clearly non-causal explanation only when an explain module exists."""
    section_title("🔬 Model Explanation (not causal)")
    try:
        import fraud_ml.models.explain as explain_module  # type: ignore
    except ImportError:
        st.caption(
            "Model explanation (not causal) is not available in this deployment. "
            "The score above is a model output only — it is neither a causal finding nor "
            "evidence of fraud. Install the optional explanation module to enable attributions."
        )
        return

    st.caption(
        "Feature attributions below describe model output only — they are **not causal** "
        "explanations and must not be used as evidence of wrongdoing."
    )
    explain_fn = getattr(explain_module, "explain_transactions", None)
    if not callable(explain_fn) or predictor is None:
        st.caption(
            "No supported local explanation entry point is available for this artifact; "
            "no attributions are shown."
        )
        return

    try:
        from fraud_ml.features.engineering import engineer_features

        from src.data_loader import load_data

        reference = load_data()
        background = engineer_features(reference.head(min(100, len(reference))))
        transactions = engineer_features(pd.DataFrame([row]))
        explanation = explain_fn(
            pipeline=predictor.pipeline,
            background=background,
            transactions=transactions,
            max_explain=1,
        )
        payload = explanation.as_dict() if hasattr(explanation, "as_dict") else dict(explanation)
        if not payload.get("available", False):
            st.caption(f"Model explanation (not causal) unavailable: {payload.get('note') or payload.get('reason')}.")
            return
        importance = payload.get("global_importance", [])
        if importance:
            names = [str(item.get("feature", "")) for item in importance]
            values = [float(item.get("mean_abs_shap", 0.0)) for item in importance]
            fig = go.Figure(go.Bar(
                x=values, y=names, orientation="h", marker_color="#6C63FF",
                hovertemplate="<b>%{y}</b><br>mean |SHAP|: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#1A1D27",
                font=dict(color="#FAFAFA"), height=280,
                margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        local = payload.get("local_explanations", [])
        st.json(local[0] if local else payload)
    except Exception:
        st.caption(
            "Model explanation (not causal) could not be computed for this transaction; "
            "no attributions are shown."
        )


def _record_prediction(result: dict, source: str, amount: float) -> None:
    """Store only non-sensitive scoring metadata in session state."""
    now = pd.Timestamp.now().strftime("%H:%M:%S")
    probability = float(result.get("fraud_probability", 0.0))
    decision = str(result.get("decision", "APPROVE"))
    risk = str(result.get("risk_level", "LOW"))
    version = str(result.get("model_version", "unknown"))
    short_id = f"…{str(result.get('transaction_id', ''))[-6:]}"

    recent = {
        "Time": now,
        "Transaction": short_id,
        "Fraud Probability": f"{probability * 100:.1f}%",
        "Risk": risk,
        "Decision": decision,
        "Model Version": version,
        "Source": source,
    }
    st.session_state.setdefault("recent_predictions", []).append(recent)

    # Legacy shape retained so the 📋 Summary page keeps working.
    legacy = {
        "Timestamp": now,
        "Amount": f"${amount:,.2f}",
        "Verdict": "🚨 REVIEW" if decision == "REVIEW" else "✅ APPROVE",
        "Fraud Score": f"{probability * 100:.1f}%",
        "Risk": risk,
        "Decision": decision,
        "Model Version": version,
        "Transaction": short_id,
    }
    st.session_state.setdefault("pred_history", []).insert(0, legacy)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔮 Live Transaction Predictor</h1>
  <p>Enter transaction details to get an instant <strong style="color:#FF4B6E;">fraud probability</strong>,
     risk level, decision, and model version from the persisted production pipeline.</p>
</div>
""", unsafe_allow_html=True)

predictor, predictor_error = load_predictor(str(settings.artifact_path))
api_url = os.getenv("FRAUD_API_URL", "").strip()

if predictor_error:
    st.warning(f"⚠️ {predictor_error}", icon="🚨")

# ── Model info banner ─────────────────────────────────────────────────────────
if predictor is not None:
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("🏆 Active Model", predictor.model_name)
    col_info2.metric("🎚️ Model Threshold", f"{predictor.threshold:.3f}")
    col_info3.metric("🧬 Model Version", predictor.model_version)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar: presets, API option, history ────────────────────────────────────
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
    st.markdown("### 🌐 Scoring Source")
    if api_url:
        use_api = st.checkbox("Use FastAPI endpoint (FRAUD_API_URL)", value=False)
        st.caption(api_url)
    else:
        use_api = False
        st.caption("FRAUD_API_URL is unset — scoring locally with the trusted artifact.")
    st.markdown("---")
    st.markdown("### 📜 Prediction History")
    history = st.session_state.get("pred_history", [])
    if history:
        st.markdown(f"**{len(history)} predictions made**")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["pred_history"] = []
            st.session_state["recent_predictions"] = []
            st.rerun()
    else:
        st.caption("No predictions yet.")

PRESETS = {
    "🟢 Typical Legitimate": dict(amt=89.99, pcd="W", c4="visa", c6="debit", email="gmail.com", device="desktop", card1=9500, C1=1, C2=1),
    "🔴 Likely Fraud": dict(amt=892.50, pcd="C", c4="mastercard", c6="credit", email="anonymous.com", device="mobile", card1=3200, C1=850, C2=720),
    "🟡 Borderline Case": dict(amt=245.00, pcd="H", c4="discover", c6="debit", email="yahoo.com", device="desktop", card1=6800, C1=45, C2=30),
}
p = PRESETS.get(preset, {})

# ── Input form ────────────────────────────────────────────────────────────────
section_title("📝 Transaction Details")
with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🆔 Transaction Info**")
        tx_id = st.text_input("Transaction ID", value=f"TX-{int(time.time())}")
        TransactionAmt = st.number_input("Amount ($)", min_value=0.01,
                                         value=float(p.get("amt", 120.00)), step=0.01)
        ProductCD = st.selectbox("Product Code", ["W", "H", "C", "S", "R"],
                                 index=["W", "H", "C", "S", "R"].index(p.get("pcd", "W")))
        card4 = st.selectbox("Card Type", ["visa", "mastercard", "discover", "amex"],
                             index=["visa", "mastercard", "discover", "amex"].index(p.get("c4", "visa")))
    with col2:
        st.markdown("**🌐 Identity Signals**")
        card6 = st.selectbox("Card Category", ["debit", "credit"],
                             index=["debit", "credit"].index(p.get("c6", "debit")))
        P_emaildomain = st.selectbox(
            "Purchaser Email Domain",
            ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", "unknown"],
            index=["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", "unknown"].index(p.get("email", "gmail.com")),
        )
        DeviceType = st.selectbox("Device Type", ["desktop", "mobile", "unknown"],
                                  index=["desktop", "mobile", "unknown"].index(p.get("device", "desktop")))
    with col3:
        st.markdown("**🔢 Behaviour Counts**")
        card1 = st.number_input("Card1 (bank code)", min_value=1000, max_value=18000,
                                value=int(p.get("card1", 9500)))
        C1 = st.number_input("C1 (address-match count)", min_value=0, max_value=3000,
                             value=int(p.get("C1", 1)))
        C2 = st.number_input("C2 (card count)", min_value=0, max_value=3000,
                             value=int(p.get("C2", 1)))

    submitted = st.form_submit_button("🔍 Run Prediction", type="primary", use_container_width=True)

# ── Prediction logic ──────────────────────────────────────────────────────────
if submitted:
    clean_id = tx_id.strip() or f"TX-{int(time.time())}"
    raw_values: dict[str, Any] = {column: None for column in RAW_COLUMNS}
    raw_values.update({
        "TransactionAmt": float(TransactionAmt),
        "TransactionDT": 43200,  # relative noon default, not a calendar timestamp
        "ProductCD": ProductCD,
        "card1": int(card1),
        "card4": card4,
        "card6": card6,
        "P_emaildomain": P_emaildomain,
        "DeviceType": DeviceType,
        "C1": int(C1),
        "C2": int(C2),
        "C6": 0,
        "C11": 0,
        "V95": 0,
        "V96": 0,
        "V97": 0,
    })
    # Strict schema: transaction_id + RAW_COLUMNS only (no target or PII beyond the form).
    row = {"transaction_id": clean_id, **{column: raw_values[column] for column in RAW_COLUMNS}}

    result: dict | None = None
    source = "local"
    if use_api and api_url:
        result = _score_via_api(row, api_url)
        source = "api"
        if result is None:
            st.caption("FastAPI endpoint was unavailable; fell back to the local trusted predictor.")
            source = "local (API fallback)"

    if result is None:
        if predictor is None:
            st.error("No local predictor is available. Train a model on the 🤖 Model Training page "
                     "or configure a reachable FRAUD_API_URL.")
            st.stop()
        try:
            scored = predictor.predict([row])
            result = dict(scored[0])
        except Exception as exc:
            st.error(f"Prediction rejected by the strict schema: {type(exc).__name__}: {exc}")
            st.stop()

    try:
        score = float(result["fraud_probability"]) * 100
    except (KeyError, TypeError, ValueError):
        st.error("The scoring response did not contain a valid fraud probability.")
        st.stop()
    risk = str(result.get("risk_level", "LOW")).upper()
    decision = str(result.get("decision", "APPROVE")).upper()
    version = str(result.get("model_version", "unknown"))
    risk_color = {"HIGH": "#FF4B6E", "MEDIUM": "#FFC107"}.get(risk, "#00D4AA")
    risk_label = {"HIGH": "🚨 HIGH RISK", "MEDIUM": "⚠️ MEDIUM RISK"}.get(risk, "✅ LOW RISK")
    flagged = decision == "REVIEW"

    st.markdown("---")
    section_title("🧾 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns([1, 1.5, 1.5])
    with res_col1:
        box_class = "fraud-box" if flagged else "safe-box"
        label_txt = "🚨 REVIEW" if flagged else "✅ APPROVE"
        label_col = "#FF4B6E" if flagged else "#6C63FF"
        st.markdown(f"""
        <div class="{box_class}">
          <p class="prob-label" style="color:{label_col};">{label_txt}</p>
          <p style="color:{label_col};font-size:2.2rem;font-weight:800;margin:0.25rem 0;">{score:.1f}%</p>
          <p style="color:#8B8FA8;margin:0;">Fraud Probability</p>
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
                "bar": {"color": risk_color},
                "bgcolor": "#1A1D27",
                "steps": [
                    {"range": [0, 40], "color": "rgba(0,212,170,0.15)"},
                    {"range": [40, 70], "color": "rgba(255,193,7,0.15)"},
                    {"range": [70, 100], "color": "rgba(255,75,110,0.15)"},
                ],
                "threshold": {"line": {"color": "#FF4B6E", "width": 3},
                              "value": settings.decision_threshold * 100},
            },
        ))
        gauge.update_layout(
            paper_bgcolor="#0E1117", font_color="#FAFAFA",
            height=260, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(gauge, use_container_width=True)

    with res_col3:
        st.markdown("""
        <div class="card card-accent">
          <h4 style="color:#00D4AA;margin-top:0;">🧾 Scoring Metadata</h4>
        """, unsafe_allow_html=True)
        st.markdown(f"- **Probability:** {score:.2f}%")
        st.markdown(f"- **Risk level:** {risk}")
        st.markdown(f"- **Decision:** {decision}")
        st.markdown(f"- **Model version:** `{version}`")
        st.markdown(f"- **Scored by:** {source}")
        st.markdown("</div>", unsafe_allow_html=True)

    _render_explanation(predictor, row, result)

    with st.expander("🔎 Input Schema (field names only — values are not logged)"):
        st.json({
            "transaction_id": clean_id,
            "strict_schema_validated": True,
            "fields_submitted": [column for column in RAW_COLUMNS if row[column] is not None],
        })

    _record_prediction(result, source, float(TransactionAmt))
    st.success("Scoring result added to recent predictions.")

# ── Prediction history table ──────────────────────────────────────────────────
if st.session_state.get("pred_history"):
    st.markdown("---")
    section_title("📜 Prediction History")
    hist_df = pd.DataFrame(st.session_state["pred_history"])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

footer()
