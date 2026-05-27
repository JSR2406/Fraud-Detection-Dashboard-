"""
src/styles.py
Centralised CSS for the Fraud Detection Dashboard.
Import and call inject_global_css() at the top of each page.
"""

import streamlit as st

# ── Palette (mirrors visualizations.py) ─────────────────────────────────────
PALETTE = {
    "fraud":   "#FF4B6E",
    "legit":   "#6C63FF",
    "accent":  "#00D4AA",
    "warn":    "#FFC107",
    "bg":      "#0E1117",
    "surface": "#1A1D27",
    "surface2":"#12151F",
    "border":  "rgba(108,99,255,0.25)",
    "text":    "#FAFAFA",
    "muted":   "#8B8FA8",
}

GLOBAL_CSS = """
<style>
/* ── Google Font ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #12151F; }
::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.5); border-radius: 3px; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1117 0%, #12151F 100%) !important;
    border-right: 1px solid rgba(108,99,255,0.15) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #6C63FF;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* ── Main content background ─────────────────────────────── */
.stApp { background: #0E1117; }

/* ── Metric cards ────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1A1D27 0%, #12151F 100%);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 16px;
    padding: 1.1rem 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 8px 32px rgba(108,99,255,0.2);
    transform: translateY(-2px);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #8B8FA8 !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #FAFAFA !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF, #5A52E0) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 15px rgba(108,99,255,0.4) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108,99,255,0.6) !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    border-radius: 8px 8px 0 0;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    transition: color 0.2s ease;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #6C63FF !important;
    border-bottom: 2px solid #6C63FF !important;
}

/* ── Expander ────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: linear-gradient(135deg, #1A1D27, #12151F);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 12px;
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #FAFAFA;
}

/* ── Info / Warning / Error boxes ────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
}

/* ── Dataframe ───────────────────────────────────────────── */
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrameGlide"] {
    border-radius: 12px;
    border: 1px solid rgba(108,99,255,0.2);
}

/* ── Select / Slider ─────────────────────────────────────── */
[data-testid="stSlider"] .stSlider > div > div > div {
    background: #6C63FF;
}

/* ── Shared utility classes ──────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0E1117 0%, #1A1D27 50%, #0d1020 100%);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 40px rgba(108,99,255,0.12), inset 0 1px 0 rgba(255,255,255,0.04);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -20%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse, rgba(108,99,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg, #6C63FF, #00D4AA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -0.03em; line-height: 1.1;
}
.hero p { color: #8B8FA8; font-size: 1.05rem; margin-top: 0.6rem; line-height: 1.6; }
.hero .hero-sub { color: #8B8FA8; }

.section-title {
    font-size: 1.2rem; font-weight: 700; color: #6C63FF;
    border-left: 4px solid #6C63FF; padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0; letter-spacing: -0.01em;
}

.badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.4);
    border-radius: 8px;
    padding: 0.25rem 0.75rem;
    font-size: 0.82rem; color: #6C63FF;
    margin: 0.2rem; font-weight: 500;
    transition: background 0.2s;
}
.badge:hover { background: rgba(108,99,255,0.25); }

.badge-green {
    background: rgba(0,212,170,0.15);
    border-color: rgba(0,212,170,0.4);
    color: #00D4AA;
}
.badge-red {
    background: rgba(255,75,110,0.15);
    border-color: rgba(255,75,110,0.4);
    color: #FF4B6E;
}

.card {
    background: linear-gradient(135deg, #1A1D27 0%, #12151F 100%);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    border-color: rgba(108,99,255,0.4);
    box-shadow: 0 8px 32px rgba(108,99,255,0.1);
}

.card-accent {
    border-left: 4px solid #6C63FF;
}

.tag {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.35);
    border-radius: 6px;
    padding: 0.2rem 0.65rem;
    font-size: 0.78rem; color: #6C63FF;
    margin: 0.2rem; font-weight: 500;
}

.fraud-box {
    background: linear-gradient(135deg, rgba(255,75,110,0.15), rgba(255,75,110,0.05));
    border: 1px solid #FF4B6E;
    border-radius: 16px; padding: 1.5rem; text-align: center;
    animation: pulse-red 2s infinite;
}
.safe-box {
    background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(108,99,255,0.05));
    border: 1px solid #6C63FF;
    border-radius: 16px; padding: 1.5rem; text-align: center;
    animation: pulse-blue 2s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,75,110,0.3); }
    50%       { box-shadow: 0 0 0 8px rgba(255,75,110,0); }
}
@keyframes pulse-blue {
    0%, 100% { box-shadow: 0 0 0 0 rgba(108,99,255,0.3); }
    50%       { box-shadow: 0 0 0 8px rgba(108,99,255,0); }
}
.prob-label { font-size: 2.5rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }

.stat-row {
    display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;
}
.stat-pill {
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 50px;
    padding: 0.4rem 1rem;
    font-size: 0.85rem; color: #FAFAFA;
    font-weight: 500;
}

.footer {
    text-align: center;
    color: #4A4E6A;
    font-size: 0.78rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid rgba(108,99,255,0.1);
    margin-top: 2rem;
}
</style>
"""


def inject_global_css():
    """Inject the shared CSS into the current Streamlit page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def section_title(text: str):
    """Render a styled section heading."""
    st.markdown(f'<p class="section-title">{text}</p>', unsafe_allow_html=True)


def footer():
    """Render the shared footer."""
    st.markdown(
        '<div class="footer">© 2024 Janmejay Singh Rathore &nbsp;·&nbsp; '
        'IEEE-CIS Fraud Detection &nbsp;·&nbsp; Built with ❤️ using Streamlit</div>',
        unsafe_allow_html=True,
    )
