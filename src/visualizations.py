"""
src/visualizations.py
Reusable Plotly chart builders for the Fraud Detection Dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = {
    "fraud":    "#FF4B6E",
    "legit":    "#6C63FF",
    "accent":   "#00D4AA",
    "bg":       "#0E1117",
    "surface":  "#1A1D27",
    "text":     "#FAFAFA",
}

_LAYOUT = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["surface"],
    font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=30, t=50, b=40),
)


def class_distribution_pie(dist: dict) -> go.Figure:
    labels = list(dist.keys())
    values = list(dist.values())
    colors = [PALETTE["legit"], PALETTE["fraud"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color=PALETTE["bg"], width=2)),
        hole=0.55,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>",
    ))
    fig.update_layout(title="Class Distribution", **_LAYOUT,
                      legend=dict(orientation="h", y=-0.1))
    return fig


def transaction_amount_hist(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for cls, color, name in [(0, PALETTE["legit"], "Legitimate"), (1, PALETTE["fraud"], "Fraud")]:
        sub = df[df["isFraud"] == cls]["TransactionAmt"]
        fig.add_trace(go.Histogram(
            x=np.log1p(sub), name=name, marker_color=color,
            opacity=0.75, nbinsx=60,
            hovertemplate=f"<b>{name}</b><br>log(Amt+1): %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        ))
    fig.update_layout(barmode="overlay", title="Transaction Amount Distribution (log scale)",
                      xaxis_title="log(TransactionAmt + 1)", yaxis_title="Count", **_LAYOUT)
    return fig


def hourly_fraud_rate(df: pd.DataFrame) -> go.Figure:
    if "hour" not in df.columns:
        df = df.copy()
        df["hour"] = (df["TransactionDT"] // 3600) % 24
    grp = df.groupby("hour")["isFraud"].mean().reset_index()
    grp.columns = ["Hour", "Fraud Rate"]
    fig = px.bar(grp, x="Hour", y="Fraud Rate", color="Fraud Rate",
                 color_continuous_scale=["#6C63FF", "#FF4B6E"],
                 title="Hourly Fraud Rate")
    fig.update_layout(**_LAYOUT)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "TransactionID"][:15]
    corr = df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(title="Feature Correlation Heatmap", **_LAYOUT)
    return fig


def feature_importance_bar(importance: dict, title: str = "Feature Importance") -> go.Figure:
    items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    names, vals = zip(*items) if items else ([], [])
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=vals, colorscale=[[0, PALETTE["legit"]], [1, PALETTE["fraud"]]]),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Importance",
                      yaxis=dict(autorange="reversed"), **_LAYOUT)
    return fig


def confusion_matrix_heatmap(cm: np.ndarray, model_name: str) -> go.Figure:
    labels = ["Legitimate", "Fraud"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, PALETTE["surface"]], [1, PALETTE["fraud"]]],
        text=cm, texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(title=f"Confusion Matrix — {model_name}",
                      xaxis_title="Predicted", yaxis_title="Actual", **_LAYOUT)
    return fig


def roc_comparison(results: dict, y_test: np.ndarray) -> go.Figure:
    from sklearn.metrics import roc_curve
    fig = go.Figure()
    colors = [PALETTE["legit"], PALETTE["accent"], PALETTE["fraud"]]
    for (name, res), color in zip(results.items(), colors):
        model  = res["model"]
        scaler = res["scaler"]
        # We need X_test — callers should pass it; skip if not possible
        if model is None:
            continue
        auc = res["metrics"]["roc_auc"]
        fig.add_trace(go.Scatter(name=f"{name} (AUC={auc:.3f})",
                                  mode="lines", line=dict(color=color, width=2)))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="gray"))
    fig.update_layout(title="ROC Curve Comparison", **_LAYOUT,
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig


def model_metrics_bar(results: dict) -> go.Figure:
    metric_names = ["accuracy", "roc_auc", "avg_prec"]
    labels       = ["Accuracy", "ROC-AUC", "Avg Precision"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=labels)
    colors = [PALETTE["legit"], PALETTE["accent"], PALETTE["fraud"]]

    for col_idx, (metric, label) in enumerate(zip(metric_names, labels), start=1):
        model_names = list(results.keys())
        vals = [results[m]["metrics"].get(metric, 0) for m in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=vals, marker_color=colors,
                   name=label, showlegend=(col_idx == 1),
                   hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.4f}}<extra></extra>"),
            row=1, col=col_idx,
        )

    fig.update_layout(title="Model Comparison", **_LAYOUT, showlegend=False)
    return fig
