"""
pages/4_ℹ️_About.py  — Project information page (redesigned).
"""

import streamlit as st
from src.styles import inject_global_css, section_title, footer

st.set_page_config(page_title="About | Fraud Detection", page_icon="ℹ️", layout="wide")
inject_global_css()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>ℹ️ About This Project</h1>
  <p>An end-to-end machine learning dashboard for financial transaction fraud detection,
     built on the <strong style="color:#00D4AA;">IEEE-CIS Fraud Detection</strong> dataset
     from Kaggle and powered by the <code>fraud_ml</code> platform modules for explicit
     ingestion, validation, chronological evaluation, threshold selection, and trusted
     inference. Interactive visualisations are exploratory; performance figures are only
     shown after you run training on your own data.</p>
</div>
""", unsafe_allow_html=True)

# ── Developer + Dataset ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
      <h4 style="color:#00D4AA;margin-top:0;">👤 Developer</h4>
      <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">
        <div style="width:60px;height:60px;border-radius:50%;
                    background:linear-gradient(135deg,#6C63FF,#00D4AA);
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.8rem;flex-shrink:0;">J</div>
        <div>
          <p style="margin:0;font-size:1.1rem;font-weight:700;">Janmejay Singh Rathore</p>
          <p style="margin:0;color:#8B8FA8;font-size:0.88rem;">Machine Learning Engineer</p>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:0.5rem;">
        <div style="display:flex;align-items:center;gap:0.6rem;font-size:0.9rem;">
          <span>📧</span>
          <a href="mailto:janmejaysingh2406@gmail.com" style="color:#6C63FF;">
            janmejaysingh2406@gmail.com</a>
        </div>
        <div style="display:flex;align-items:center;gap:0.6rem;font-size:0.9rem;">
          <span>🐙</span>
          <a href="https://github.com/JSR2406" target="_blank" style="color:#6C63FF;">
            github.com/JSR2406</a>
        </div>
        <div style="display:flex;align-items:center;gap:0.6rem;font-size:0.9rem;">
          <span>🌐</span>
          <a href="https://fraudexplainableai.streamlit.app/" target="_blank" style="color:#6C63FF;">
            Live Demo on Streamlit Cloud</a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <h4 style="color:#00D4AA;margin-top:0;">🗂️ Dataset</h4>
      <table width="100%" style="color:#FAFAFA;font-size:0.88rem;border-collapse:collapse;">
        <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
          <td style="padding:0.4rem 0;color:#8B8FA8;">Source</td>
          <td>Kaggle — IEEE-CIS Fraud Detection</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
          <td style="padding:0.4rem 0;color:#8B8FA8;">Transactions</td>
          <td>590,540 rows × 394 columns</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
          <td style="padding:0.4rem 0;color:#8B8FA8;">Identity</td>
          <td>144,233 rows × 41 columns</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
          <td style="padding:0.4rem 0;color:#8B8FA8;">Target</td>
          <td><code>isFraud</code> — binary (~3.5% positive)</td>
        </tr>
        <tr>
          <td style="padding:0.4rem 0;color:#8B8FA8;">Data mode</td>
          <td>Explicit: <code>real</code> or <code>demo</code> via FRAUD_DATA_MODE. Demo = synthetic smoke-test data, never a real result.</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
      <h4 style="color:#00D4AA;margin-top:0;">🤖 Model Candidates</h4>
      <table width="100%" style="color:#FAFAFA;font-size:0.88rem;border-collapse:collapse;">
        <thead>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
            <th style="padding:0.4rem 0;text-align:left;color:#8B8FA8;">Candidate</th>
            <th style="text-align:left;color:#8B8FA8;">Role</th>
          </tr>
        </thead>
        <tbody>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="padding:0.5rem 0;">⭐ LightGBM</td>
            <td>Gradient-boosted trees (when installed)</td>
          </tr>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="padding:0.5rem 0;">🌲 Random Forest</td>
            <td>Ensemble baseline with feature importances</td>
          </tr>
          <tr>
            <td style="padding:0.5rem 0;">📉 Logistic Regression</td>
            <td>Fast, scaled linear baseline</td>
          </tr>
        </tbody>
      </table>
      <p style="color:#8B8FA8;font-size:0.78rem;margin-top:0.75rem;">
        Selection is made on <strong>validation average precision</strong>; final test metrics
        are reported once after selection. No fixed AUC values are claimed here — see the
        🤖 Model Training page after running training on your configured data.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <h4 style="color:#00D4AA;margin-top:0;">🛠️ Tech Stack</h4>
    """, unsafe_allow_html=True)
    techs = [
        ("Python 3.11", "🐍"), ("Streamlit", "🎈"), ("LightGBM", "⚡"),
        ("scikit-learn", "🔬"), ("Pandas", "🐼"), ("NumPy", "🔢"),
        ("Plotly", "📊"), ("FastAPI (optional)", "🚀"),
    ]
    tags_html = " ".join([f'<span class="tag">{e} {t}</span>' for t, e in techs])
    st.markdown(tags_html + "</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card card-accent">
      <h4 style="color:#6C63FF;margin-top:0;">📈 Evaluation &amp; Configuration</h4>
      <ul style="color:#8B8FA8;font-size:0.88rem;padding-left:1.2rem;line-height:1.7;margin:0;">
        <li><code>fraud_ml.data.ingestion</code> — explicit real/demo loading; no silent fallback.</li>
        <li><code>fraud_ml.data.validation</code> — schema and quality checks.</li>
        <li><code>fraud_ml.data.splitting</code> — chronological train/validation/test split.</li>
        <li><code>fraud_ml.features</code> — row-local engineering and leak-safe preprocessing.</li>
        <li><code>fraud_ml.models</code> — training, evaluation and threshold optimisation.</li>
        <li><code>fraud_ml.inference</code> — strict-schema, trusted-artifact scoring.</li>
        <li>Decision threshold, train/validation fractions and cost assumptions are configurable
            via <code>fraud_ml.config.settings</code> and environment variables.</li>
        <li>No performance numbers are hard-coded: results come only from your own training run.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# ── Project timeline ──────────────────────────────────────────────────────────
st.markdown("---")
section_title("🗓️ Project Timeline")
st.markdown("""
<div class="card">
<div style="display:flex;flex-direction:column;gap:0;position:relative;padding-left:2rem;">
  <div style="position:absolute;left:0.65rem;top:0;bottom:0;width:2px;background:rgba(108,99,255,0.25);"></div>
  <div style="position:relative;padding-bottom:1.25rem;">
    <div style="position:absolute;left:-1.6rem;top:3px;width:12px;height:12px;border-radius:50%;background:#6C63FF;"></div>
    <p style="margin:0;font-size:0.78rem;color:#8B8FA8;">Phase 1</p>
    <p style="margin:0;font-weight:600;">📥 Data Ingestion & EDA</p>
    <p style="margin:0;font-size:0.88rem;color:#8B8FA8;">Explicit real/demo ingestion with validation and quality reports.</p>
  </div>
  <div style="position:relative;padding-bottom:1.25rem;">
    <div style="position:absolute;left:-1.6rem;top:3px;width:12px;height:12px;border-radius:50%;background:#00D4AA;"></div>
    <p style="margin:0;font-size:0.78rem;color:#8B8FA8;">Phase 2</p>
    <p style="margin:0;font-weight:600;">⚙️ Feature Engineering & Preprocessing</p>
    <p style="margin:0;font-size:0.88rem;color:#8B8FA8;">Time-based features, log-transforms, categorical encoding pipeline.</p>
  </div>
  <div style="position:relative;padding-bottom:1.25rem;">
    <div style="position:absolute;left:-1.6rem;top:3px;width:12px;height:12px;border-radius:50%;background:#FFC107;"></div>
    <p style="margin:0;font-size:0.78rem;color:#8B8FA8;">Phase 3</p>
    <p style="margin:0;font-weight:600;">🤖 Model Training & Comparison</p>
    <p style="margin:0;font-size:0.88rem;color:#8B8FA8;">Chronological evaluation, validation-based threshold selection, persisted candidate artifact.</p>
  </div>
  <div style="position:relative;padding-bottom:1.25rem;">
    <div style="position:absolute;left:-1.6rem;top:3px;width:12px;height:12px;border-radius:50%;background:#FF4B6E;"></div>
    <p style="margin:0;font-size:0.78rem;color:#8B8FA8;">Phase 4</p>
    <p style="margin:0;font-weight:600;">🔮 Live Prediction Interface</p>
    <p style="margin:0;font-size:0.88rem;color:#8B8FA8;">Strict-schema trusted-artifact scoring, optional FastAPI endpoint, batch CSV scoring, non-causal explanations.</p>
  </div>
  <div style="position:relative;">
    <div style="position:absolute;left:-1.6rem;top:3px;width:12px;height:12px;border-radius:50%;background:linear-gradient(135deg,#6C63FF,#00D4AA);"></div>
    <p style="margin:0;font-size:0.78rem;color:#8B8FA8;">Phase 5</p>
    <p style="margin:0;font-weight:600;">🚀 Deployed on Streamlit Cloud</p>
    <p style="margin:0;font-size:0.88rem;color:#8B8FA8;">Public demo at <a href="https://fraudexplainableai.streamlit.app/" style="color:#6C63FF;">fraudexplainableai.streamlit.app</a> · fraud_ml platform integration.</p>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

# ── Project structure ─────────────────────────────────────────────────────────
section_title("🏗️ Project Structure")
with st.expander("Show project tree"):
    st.code("""
FraudDetection_JANMEJAY_SINGH_RATHORE/
├── app.py                      ← Home dashboard + provenance banner
├── fraud_ml/                   ← New fraud platform
│   ├── config.py               ← Explicit settings (mode, paths, threshold)
│   ├── data/                   ← Ingestion, validation, chronological splitting
│   ├── features/               ← Row-local feature engineering & preprocessing
│   ├── models/                 ← Training, evaluation, threshold selection
│   ├── inference/              ← Strict-schema trusted-artifact scoring
│   └── api/ (optional)         ← FastAPI scoring service
├── src/
│   ├── data_loader.py          ← Thin wrapper over fraud_ml.data.ingestion
│   ├── visualizations.py       ← Plotly chart builders
│   └── styles.py               ← Shared CSS & utilities
└── pages/
    ├── 1_📊_EDA.py             ← Exploratory Data Analysis
    ├── 2_🤖_Model_Training.py  ← Platform training & comparison
    ├── 3_🔮_Predict.py         ← Live scoring + history
    ├── 4_ℹ️_About.py           ← This page
    ├── 5_📋_Summary.py         ← Overall summarisation
    └── 6_📦_Batch_Detection.py ← CSV batch scoring
""", language="")

footer()
