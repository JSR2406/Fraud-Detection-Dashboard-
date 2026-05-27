"""
pages/4_ℹ️_About.py
Project information page.
"""

import streamlit as st

st.set_page_config(page_title="About | Fraud Detection", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.card {
    background: linear-gradient(135deg,#1A1D27,#12151F);
    border: 1px solid rgba(108,99,255,.25); border-radius:16px;
    padding:1.5rem 2rem; margin-bottom:1rem;
}
.tag {
    display:inline-block; background:rgba(108,99,255,.15);
    border:1px solid rgba(108,99,255,.4); border-radius:6px;
    padding:.2rem .6rem; font-size:.8rem; color:#6C63FF; margin:.2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("ℹ️ About This Project")

st.markdown("""
<div class="card">
<h3 style="color:#6C63FF;">🛡️ Fraud Detection Dashboard</h3>
<p>
An end-to-end machine learning project for detecting financial transaction fraud using the 
<strong>IEEE-CIS Fraud Detection</strong> dataset from Kaggle. The dashboard provides interactive
exploration, model training with comparison metrics, and a live prediction interface.
</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
    <h4 style="color:#00D4AA;">👤 Developer</h4>
    <p><strong>Janmejay Singh Rathore</strong></p>
    <p style="color:#8B8FA8;">Machine Learning Engineer</p>
    <br>
    <p>📧 janmejaysingh2406@gmail.com</p>
    <p>🐙 <a href="https://github.com/JSR2406" style="color:#6C63FF;">github.com/JSR2406</a></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h4 style="color:#00D4AA;">🗂️ Dataset</h4>
    <ul>
    <li><strong>Source:</strong> Kaggle — IEEE-CIS Fraud Detection</li>
    <li><strong>Train transactions:</strong> 590,540 rows × 394 cols</li>
    <li><strong>Train identity:</strong> 144,233 rows × 41 cols</li>
    <li><strong>Target:</strong> isFraud (binary, ~3.5% positive rate)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h4 style="color:#00D4AA;">🤖 Models Used</h4>
    <table width="100%" style="color:#FAFAFA;">
    <tr><th>Model</th><th>Key Strength</th></tr>
    <tr><td>⭐ LightGBM</td><td>Best AUC, handles imbalance well</td></tr>
    <tr><td>🌲 Random Forest</td><td>Robust, interpretable</td></tr>
    <tr><td>📉 Logistic Regression</td><td>Fast baseline, explainable</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h4 style="color:#00D4AA;">🛠️ Tech Stack</h4>
    """, unsafe_allow_html=True)

    techs = ["Python 3.11", "Streamlit", "LightGBM", "scikit-learn",
             "Pandas", "NumPy", "Plotly", "imbalanced-learn"]
    tags = " ".join([f'<span class="tag">{t}</span>' for t in techs])
    st.markdown(tags + "</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="card">
<h4 style="color:#00D4AA;">🏗️ Project Structure</h4>

```
FraudDetection_JANMEJAY_SINGH_RATHORE/
├── app.py                    ← Main dashboard (Home)
├── requirements.txt          ← Python dependencies
├── README.md                 ← Project documentation
├── .gitignore
├── .streamlit/
│   └── config.toml           ← Theme & server config
├── src/
│   ├── __init__.py
│   ├── data_loader.py        ← Data loading & caching
│   ├── preprocessing.py      ← Feature engineering
│   ├── model_trainer.py      ← Training pipeline
│   └── visualizations.py     ← Plotly charts
├── pages/
│   ├── 1_📊_EDA.py           ← Exploratory Data Analysis
│   ├── 2_🤖_Model_Training.py← Train & compare models
│   ├── 3_🔮_Predict.py       ← Live prediction
│   └── 4_ℹ️_About.py         ← This page
├── data/
│   ├── train_transaction.csv.zip
│   └── train_identity.csv.zip
└── charts/
    └── correlation_heatmap.png
```
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h4 style="color:#00D4AA;">📈 Key Results (LightGBM on IEEE-CIS)</h4>
<table width="100%" style="color:#FAFAFA;">
<tr><th>Metric</th><th>Score</th></tr>
<tr><td>ROC-AUC</td><td>~0.98</td></tr>
<tr><td>Accuracy</td><td>~98.5%</td></tr>
<tr><td>Avg Precision</td><td>~0.82</td></tr>
</table>
<p style="color:#8B8FA8;font-size:.85rem;">* Scores on 20% held-out test set. Demo mode uses synthetic data.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
---
<center style="color:#555;font-size:.8rem;">
© 2024 Janmejay Singh Rathore · IEEE-CIS Fraud Detection · Built with Streamlit
</center>
""", unsafe_allow_html=True)
