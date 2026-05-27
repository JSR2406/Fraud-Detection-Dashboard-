# 🛡️ Fraud Detection Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraudexplainableai.streamlit.app/)

> End-to-end machine learning dashboard for detecting financial transaction fraud using the **IEEE-CIS Fraud Detection** dataset.

**Developed by:** Janmejay Singh Rathore  
**Model:** LightGBM (primary) · Random Forest · Logistic Regression

---

## 🚀 Live Demo

👉 **[Open on Streamlit Cloud](https://fraudexplainableai.streamlit.app/)**

---

## 📋 Features

| Page | Description |
|------|-------------|
| 🏠 **Home** | KPI overview, class distribution, amount histograms |
| 📊 **EDA** | Deep-dive data exploration — distributions, correlations, patterns |
| 🤖 **Model Training** | Train & compare 3 models with configurable params |
| 🔮 **Predict** | Live fraud prediction with gauge chart |
| ℹ️ **About** | Project info, tech stack, structure |

---

## 🗂️ Project Structure

```
FraudDetection_JANMEJAY_SINGH_RATHORE/
├── app.py                    ← Main dashboard (Home page)
├── requirements.txt          ← Python dependencies
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml           ← Dark purple theme
├── src/
│   ├── __init__.py
│   ├── data_loader.py        ← IEEE-CIS loader + synthetic fallback
│   ├── preprocessing.py      ← Feature engineering pipeline
│   ├── model_trainer.py      ← LightGBM / RF / LR training
│   └── visualizations.py     ← Plotly chart builders
├── pages/
│   ├── 1_📊_EDA.py
│   ├── 2_🤖_Model_Training.py
│   ├── 3_🔮_Predict.py
│   └── 4_ℹ️_About.py
├── data/
│   ├── train_transaction.csv.zip   ← IEEE-CIS (not tracked by git)
│   └── train_identity.csv.zip
└── charts/
    └── correlation_heatmap.png
```

---

## 🛠️ Installation & Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/JSR2406/Fraud-Detection-Dashboard-.git
cd Fraud-Detection-Dashboard-

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Add IEEE-CIS data to data/ folder
# Download from: https://www.kaggle.com/c/ieee-fraud-detection/data

# 4. Run the dashboard
streamlit run app.py
```

> **Note:** Without the IEEE-CIS CSV files, the app automatically uses a synthetic demo dataset so all features remain functional.

---

## 🤖 Models & Performance

| Model | ROC-AUC | Accuracy | Avg Precision |
|-------|---------|----------|---------------|
| ⭐ LightGBM | ~0.980 | ~98.5% | ~0.82 |
| 🌲 Random Forest | ~0.960 | ~97.8% | ~0.72 |
| 📉 Logistic Regression | ~0.870 | ~96.5% | ~0.55 |

*Scores on 20% held-out test set from IEEE-CIS data.*

---

## 📦 Tech Stack

- **Python 3.11**
- **Streamlit** — Dashboard framework
- **LightGBM** — Gradient boosting (primary model)
- **scikit-learn** — Random Forest, Logistic Regression, metrics
- **Pandas / NumPy** — Data manipulation
- **Plotly** — Interactive visualizations
- **imbalanced-learn** — Class imbalance handling

---

## 📊 Dataset

- **Source:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
- **Train transactions:** 590,540 rows × 394 features
- **Train identity:** 144,233 rows × 41 features  
- **Target:** `isFraud` (binary; ~3.5% positive rate)

---

## 📄 License

MIT License — © 2024 Janmejay Singh Rathore