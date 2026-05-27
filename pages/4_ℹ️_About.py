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
     from Kaggle with production-grade ML pipelines and interactive visualisations.</p>
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
          <td style="padding:0.4rem 0;color:#8B8FA8;">Demo fallback</td>
          <td>5,000-row synthetic dataset</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
      <h4 style="color:#00D4AA;margin-top:0;">🤖 Models</h4>
      <table width="100%" style="color:#FAFAFA;font-size:0.88rem;border-collapse:collapse;">
        <thead>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
            <th style="padding:0.4rem 0;text-align:left;color:#8B8FA8;">Model</th>
            <th style="text-align:left;color:#8B8FA8;">Strength</th>
            <th style="text-align:left;color:#8B8FA8;">AUC*</th>
          </tr>
        </thead>
        <tbody>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="padding:0.5rem 0;">⭐ LightGBM</td>
            <td>Best AUC, handles imbalance</td>
            <td style="color:#00D4AA;font-weight:700;">~0.98</td>
          </tr>
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="padding:0.5rem 0;">🌲 Random Forest</td>
            <td>Robust, interpretable</td>
            <td style="color:#6C63FF;font-weight:700;">~0.95</td>
          </tr>
          <tr>
            <td style="padding:0.5rem 0;">📉 Logistic Reg.</td>
            <td>Fast baseline, explainable</td>
            <td style="color:#FFC107;font-weight:700;">~0.88</td>
          </tr>
        </tbody>
      </table>
      <p style="color:#8B8FA8;font-size:0.78rem;margin-top:0.75rem;">* On 20% held-out test set. Demo uses synthetic data.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <h4 style="color:#00D4AA;margin-top:0;">🛠️ Tech Stack</h4>
    """, unsafe_allow_html=True)
    techs = [
        ("Python 3.11", "🐍"), ("Streamlit", "🎈"), ("LightGBM", "⚡"),
        ("scikit-learn", "🔬"), ("Pandas", "🐼"), ("NumPy", "🔢"),
        ("Plotly", "📊"), ("imbalanced-learn", "⚖️"),
    ]
    tags_html = " ".join([f'<span class="tag">{e} {t}</span>' for t, e in techs])
    st.markdown(tags_html + "</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <h4 style="color:#00D4AA;margin-top:0;">📈 Key Results</h4>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.75rem;text-align:center;">
        <div style="background:rgba(108,99,255,0.1);border-radius:10px;padding:0.75rem;">
          <p style="font-size:1.4rem;font-weight:800;color:#6C63FF;margin:0;">0.98</p>
          <p style="color:#8B8FA8;font-size:0.78rem;margin:0;">ROC-AUC</p>
        </div>
        <div style="background:rgba(0,212,170,0.1);border-radius:10px;padding:0.75rem;">
          <p style="font-size:1.4rem;font-weight:800;color:#00D4AA;margin:0;">98.5%</p>
          <p style="color:#8B8FA8;font-size:0.78rem;margin:0;">Accuracy</p>
        </div>
        <div style="background:rgba(255,193,7,0.1);border-radius:10px;padding:0.75rem;">
          <p style="font-size:1.4rem;font-weight:800;color:#FFC107;margin:0;">0.82</p>
          <p style="color:#8B8FA8;font-size:0.78rem;margin:0;">Avg Precision</p>
        </div>
      </div>
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
    <p style="margin:0;font-size:0.88rem;color:#8B8FA8;">Loaded IEEE-CIS dataset, explored fraud patterns, built synthetic fallback.</p>
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
    <p style="margin:0;font-size:0.88rem;color:#8B8FA8;">LightGBM, Random Forest, Logistic Regression — trained with class-weight balancing.</p>
  </div>
  <div style="position:relative;padding-bottom:1.25rem;">
    <div style="position:absolute;left:-1.6rem;top:3px;width:12px;height:12px;border-radius:50%;background:#FF4B6E;"></div>
    <p style="margin:0;font-size:0.78rem;color:#8B8FA8;">Phase 4</p>
    <p style="margin:0;font-weight:600;">🔮 Live Prediction Interface</p>
    <p style="margin:0;font-size:0.88rem;color:#8B8FA8;">Real-time scoring with gauge chart, feature contributions & prediction history.</p>
  </div>
  <div style="position:relative;">
    <div style="position:absolute;left:-1.6rem;top:3px;width:12px;height:12px;border-radius:50%;background:linear-gradient(135deg,#6C63FF,#00D4AA);"></div>
    <p style="margin:0;font-size:0.78rem;color:#8B8FA8;">Phase 5</p>
    <p style="margin:0;font-weight:600;">🚀 Deployed on Streamlit Cloud</p>
    <p style="margin:0;font-size:0.88rem;color:#8B8FA8;">Public demo at <a href="https://fraudexplainableai.streamlit.app/" style="color:#6C63FF;">fraudexplainableai.streamlit.app</a></p>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

# ── Project structure ─────────────────────────────────────────────────────────
section_title("🏗️ Project Structure")
with st.expander("Show project tree"):
    st.code("""
FraudDetection_JANMEJAY_SINGH_RATHORE/
├── app.py                      ← Home dashboard
├── requirements.txt            ← Python dependencies
├── README.md
├── .streamlit/
│   └── config.toml             ← Dark theme & server config
├── src/
│   ├── __init__.py
│   ├── data_loader.py          ← Data loading & synthetic fallback
│   ├── preprocessing.py        ← Feature engineering pipeline
│   ├── model_trainer.py        ← LightGBM / RF / LR training
│   ├── visualizations.py       ← Plotly chart builders
│   └── styles.py               ← Shared CSS & utilities
└── pages/
    ├── 1_📊_EDA.py             ← Exploratory Data Analysis
    ├── 2_🤖_Model_Training.py  ← Train & compare models
    ├── 3_🔮_Predict.py         ← Live prediction + history
    └── 4_ℹ️_About.py           ← This page
""", language="")

footer()
