"""
pages/6_📦_Batch_Detection.py
Validate and score a batch of transactions from CSV with the trusted predictor.
"""

import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st

from fraud_ml.config import settings
from fraud_ml.features.engineering import RAW_COLUMNS, NUMERIC, CATEGORICAL
from fraud_ml.inference.predictor import FraudPredictor
from src.data_loader import dataset_source_label, is_demo_source
from src.styles import inject_global_css, section_title, footer

st.set_page_config(page_title="Batch Detection | Fraud Detection", page_icon="📦", layout="wide")
inject_global_css()

SOURCE_LABEL = dataset_source_label()
if is_demo_source(SOURCE_LABEL):
    st.warning(
        "🚧 This dashboard is running with **synthetic demo data**. Any batch scores are "
        "illustrative outputs of the current artifact, **not validated real-world fraud findings**. "
        "The artifact itself may also have been trained on demo data — verify training provenance "
        "before operational use.",
        icon="⚠️",
    )

ALLOWED_COLUMNS = ["transaction_id", *RAW_COLUMNS]
NUMERIC_RAW = [column for column in NUMERIC if column in RAW_COLUMNS]
CATEGORICAL_RAW = [column for column in CATEGORICAL if column in RAW_COLUMNS]


@st.cache_resource(show_spinner="Loading persisted model …")
def load_predictor(path: str) -> tuple[FraudPredictor | None, str | None]:
    artifact = Path(path)
    if not artifact.is_file():
        return None, "No persisted model artifact found. Train on the 🤖 Model Training page."
    try:
        return FraudPredictor.from_trusted_file(artifact), None
    except Exception as exc:
        return None, f"Could not load the trusted artifact: {type(exc).__name__}."


st.markdown("""
<div class="hero">
  <h1>📦 Batch Fraud Detection</h1>
  <p>Upload a CSV, validate it against the strict <code>fraud_ml</code> input schema, and score
     rows with the same trusted artifact used by live prediction. Only scoring metadata is
     returned — no sensitive fields are logged.</p>
</div>
""", unsafe_allow_html=True)

predictor, predictor_error = load_predictor(str(settings.artifact_path))
if predictor_error:
    st.warning(f"⚠️ {predictor_error}", icon="🚨")
else:
    assert predictor is not None
    st.markdown(
        f'<span class="badge badge-green">Active model: {predictor.model_name}</span>'
        f'<span class="badge">Version: {predictor.model_version}</span>'
        f'<span class="badge">Threshold: {predictor.threshold:.3f}</span>',
        unsafe_allow_html=True,
    )

section_title("📄 Upload Transactions")
st.caption(
    "Allowed columns: `transaction_id` plus the raw feature fields "
    "(`TransactionAmt`, `TransactionDT`, `ProductCD`, card/C/V fields, `P_emaildomain`, `DeviceType`)."
)
upload = st.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=False)

auto_ids = st.checkbox("Generate transaction_id when the file omits it", value=True)
if upload is not None:
    try:
        uploaded = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not parse the CSV: {type(exc).__name__}: {exc}")
        st.stop()

    st.success(f"Loaded {len(uploaded):,} rows × {uploaded.shape[1]} columns.")
    section_title("👀 Preview")
    st.dataframe(uploaded.head(10), use_container_width=True)

    section_title("✅ Schema Validation")
    unknown = sorted(set(uploaded.columns) - set(ALLOWED_COLUMNS))
    if unknown:
        st.error(
            f"Unexpected column(s): {unknown}. Allowed columns are: {', '.join(ALLOWED_COLUMNS)}. "
            "Remove or rename them before scoring."
        )
        st.stop()
    if "TransactionAmt" not in uploaded.columns:
        st.error("`TransactionAmt` is required by the predictor.")
        st.stop()
    if "transaction_id" not in uploaded.columns:
        if auto_ids:
            uploaded = uploaded.assign(
                transaction_id=[f"BATCH-{i + 1:06d}" for i in range(len(uploaded))]
            )
            st.info("Generated sequential `transaction_id` values for this batch.")
        else:
            st.error("`transaction_id` is required. Enable the auto-generate option above.")
            st.stop()

    with st.expander("Column check details"):
        st.markdown(f"- Recognised columns: {', '.join(map(str, uploaded.columns))}")
        st.markdown(f"- Numeric fields coerced: {', '.join(NUMERIC_RAW)}")
        st.markdown(f"- Categorical fields: {', '.join(CATEGORICAL_RAW)}")

    max_rows = min(int(settings.max_batch_size), len(uploaded))
    limit = st.number_input("Rows to score", min_value=1, max_value=max_rows,
                            value=min(100, max_rows))
    score_btn = st.button("🚀 Score Batch", type="primary", use_container_width=True)

    if score_btn:
        if predictor is None:
            st.error("No trusted predictor is available; train a model first.")
            st.stop()

        batch = uploaded.head(int(limit)).copy()
        for column in NUMERIC_RAW:
            if column in batch.columns:
                batch[column] = pd.to_numeric(batch[column], errors="coerce")
        for column in CATEGORICAL_RAW:
            if column in batch.columns:
                batch[column] = batch[column].astype("string").fillna("unknown").astype(str)

        # Strict schema: transaction_id + RAW_COLUMNS only.
        rows = [
            {"transaction_id": str(record.pop("transaction_id")),
             **{column: record.get(column) for column in RAW_COLUMNS}}
            for record in batch.to_dict("records")
        ]
        try:
            with st.spinner(f"Scoring {len(rows):,} transactions …"):
                results = predictor.predict(rows)
        except Exception as exc:
            st.error(f"Batch scoring was rejected by the predictor: {type(exc).__name__}: {exc}")
            st.stop()

        results_df = pd.DataFrame(results)
        display_df = results_df.rename(columns={
            "transaction_id": "Transaction",
            "fraud_probability": "Fraud Probability",
            "risk_level": "Risk",
            "decision": "Decision",
            "model_version": "Model Version",
        })
        display_df["Fraud Probability"] = (display_df["Fraud Probability"] * 100).round(2)
        display_df = display_df.rename(columns={"Fraud Probability": "Fraud Probability (%)"})

        section_title("📊 Batch Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows scored", len(display_df))
        c2.metric("REVIEW", int((display_df["Decision"] == "REVIEW").sum()))
        c3.metric("APPROVE", int((display_df["Decision"] == "APPROVE").sum()))
        c4.metric("High risk", int((display_df["Risk"] == "HIGH").sum()))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        section_title("📥 Download Results")
        output = io.BytesIO()
        results_df.to_csv(output, index=False)
        st.download_button(
            label="⬇️ Download scored CSV",
            data=output.getvalue(),
            file_name="fraud_batch_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Keep the Home page's recent panel current without storing whole batches.
        now = pd.Timestamp.now().strftime("%H:%M:%S")
        for result in results[:5]:
            st.session_state.setdefault("recent_predictions", []).append({
                "Time": now,
                "Transaction": f"…{str(result['transaction_id'])[-6:]}",
                "Fraud Probability": f"{float(result['fraud_probability']) * 100:.1f}%",
                "Risk": result["risk_level"],
                "Decision": result["decision"],
                "Model Version": result["model_version"],
                "Source": "batch",
            })
        st.caption("Scoring metadata from up to five rows was added to the Home page's recent panel.")

else:
    st.info("Upload a CSV above to validate and score a batch.", icon="📤")

footer()
