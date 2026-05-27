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
    "fraud":   "#FF4B6E",
    "legit":   "#6C63FF",
    "accent":  "#00D4AA",
    "warn":    "#FFC107",
    "bg":      "#0E1117",
    "surface": "#1A1D27",
    "text":    "#FAFAFA",
}

_LAYOUT = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["surface"],
    font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=30, t=50, b=40),
)

_MODEL_COLORS = [PALETTE["legit"], PALETTE["accent"], PALETTE["warn"], PALETTE["fraud"]]


def class_distribution_pie(dist: dict) -> go.Figure:
    labels = list(dist.keys())
    values = list(dist.values())
    colors = [PALETTE["legit"], PALETTE["fraud"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color=PALETTE["bg"], width=2)),
        hole=0.58,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"{values[1]:,}<br><span style='font-size:11px'>fraud</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color=PALETTE["fraud"]),
    )
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
    fig.update_layout(
        barmode="overlay",
        title="Transaction Amount Distribution (log scale)",
        xaxis_title="log(TransactionAmt + 1)", yaxis_title="Count",
        legend=dict(orientation="h", y=1.08),
        **_LAYOUT,
    )
    return fig


def hourly_fraud_rate(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    if "hour" not in df2.columns:
        df2["hour"] = (df2["TransactionDT"] // 3600) % 24
    grp = df2.groupby("hour")["isFraud"].mean().reset_index()
    grp.columns = ["Hour", "Fraud Rate"]
    fig = px.bar(grp, x="Hour", y="Fraud Rate", color="Fraud Rate",
                 color_continuous_scale=["#6C63FF", "#FFC107", "#FF4B6E"],
                 title="Hourly Fraud Rate",
                 hover_data={"Fraud Rate": ":.2%"})
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
    if not items:
        return go.Figure()
    names, vals = zip(*items)
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(
            color=vals,
            colorscale=[[0, PALETTE["legit"]], [0.5, PALETTE["warn"]], [1, PALETTE["fraud"]]],
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        **_LAYOUT,
    )
    return fig


def confusion_matrix_heatmap(cm: np.ndarray, model_name: str) -> go.Figure:
    labels = ["Legitimate", "Fraud"]
    # Normalised percentages for annotation
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annotations = [
        [f"{cm[i][j]:,}<br>({cm_norm[i][j]*100:.1f}%)" for j in range(2)]
        for i in range(2)
    ]
    fig = go.Figure(go.Heatmap(
        z=cm_norm, x=labels, y=labels,
        colorscale=[[0, PALETTE["surface"]], [1, PALETTE["fraud"]]],
        text=annotations, texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:.2%}<extra></extra>",
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted", yaxis_title="Actual",
        **_LAYOUT,
    )
    return fig


def roc_curve_comparison(results: dict, X_test: np.ndarray, y_test: np.ndarray) -> go.Figure:
    """Plot ROC curves for all trained models."""
    from sklearn.metrics import roc_curve, auc as sk_auc
    fig = go.Figure()
    for (name, res), color in zip(results.items(), _MODEL_COLORS):
        mdl    = res["model"]
        scaler = res["scaler"]
        if mdl is None:
            continue
        Xp   = scaler.transform(X_test) if scaler else X_test
        prob = mdl.predict_proba(Xp)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_val = sk_auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={auc_val:.3f})",
            line=dict(color=color, width=2.5),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))
    # Diagonal baseline
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="rgba(255,255,255,0.3)", width=1))
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(orientation="h", y=-.18),
        **_LAYOUT,
    )
    return fig


def pr_curve_comparison(results: dict, X_test: np.ndarray, y_test: np.ndarray) -> go.Figure:
    """Plot Precision-Recall curves for all trained models."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig = go.Figure()
    for (name, res), color in zip(results.items(), _MODEL_COLORS):
        mdl    = res["model"]
        scaler = res["scaler"]
        if mdl is None:
            continue
        Xp   = scaler.transform(X_test) if scaler else X_test
        prob = mdl.predict_proba(Xp)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{name} (AP={ap:.3f})",
            line=dict(color=color, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)",
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        ))
    baseline = y_test.mean() if y_test is not None else 0.035
    fig.add_shape(type="line", x0=0, y0=baseline, x1=1, y1=baseline,
                  line=dict(dash="dash", color="rgba(255,255,255,0.3)", width=1))
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall", yaxis_title="Precision",
        legend=dict(orientation="h", y=-.18),
        **_LAYOUT,
    )
    return fig


def model_metrics_bar(results: dict) -> go.Figure:
    metric_names = ["accuracy", "roc_auc", "avg_prec"]
    labels       = ["Accuracy", "ROC-AUC", "Avg Precision"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=labels)

    for col_idx, (metric, label) in enumerate(zip(metric_names, labels), start=1):
        model_names = list(results.keys())
        vals   = [results[m]["metrics"].get(metric, 0) for m in model_names]
        colors = _MODEL_COLORS[:len(model_names)]
        fig.add_trace(
            go.Bar(
                x=model_names, y=vals, marker_color=colors,
                name=label, showlegend=False,
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
                hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.4f}}<extra></extra>",
            ),
            row=1, col=col_idx,
        )
        fig.update_yaxes(range=[0, 1.05], row=1, col=col_idx)

    fig.update_layout(title="Model Comparison", **_LAYOUT, showlegend=False)
    return fig
