"""Plotly chart helpers for the trend detection demo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_pipeline import CHANNEL_NAMES, CLASSES


def trend_line_chart(series: pd.DataFrame, window_start: int, window_size: int) -> go.Figure:
    fig = go.Figure()
    t = series["display_time"]
    fig.add_trace(
        go.Scatter(
            x=t, y=series["value"], mode="lines+markers", name="value",
            line=dict(color="#1f77b4"), marker=dict(size=4),
        )
    )
    fig.add_trace(
        go.Scatter(x=t, y=series["target"], mode="lines", name="target",
                   line=dict(color="gray", dash="dash"))
    )
    fig.add_trace(
        go.Scatter(x=t, y=series["ucl"], mode="lines", name="UCL (3σ)",
                   line=dict(color="orange", dash="dot"))
    )
    fig.add_trace(
        go.Scatter(x=t, y=series["lcl"], mode="lines", name="LCL (3σ)",
                   line=dict(color="orange", dash="dot"), showlegend=False)
    )
    fig.add_trace(
        go.Scatter(x=t, y=series["critical_high"], mode="lines", name="critical (5σ)",
                   line=dict(color="red", dash="dot"))
    )
    fig.add_trace(
        go.Scatter(x=t, y=series["critical_low"], mode="lines",
                   line=dict(color="red", dash="dot"), showlegend=False)
    )

    anomalous = series[series["is_anomalous_point"]]
    if len(anomalous) > 0:
        fig.add_trace(
            go.Scatter(
                x=anomalous["display_time"], y=anomalous["value"], mode="markers", name="GT anomaly",
                marker=dict(color="red", size=8, symbol="x"),
            )
        )

    if 0 <= window_start <= len(series) - window_size:
        w_start_t = series.iloc[window_start]["display_time"]
        w_end_t = series.iloc[window_start + window_size - 1]["display_time"]
        fig.add_vrect(x0=w_start_t, x1=w_end_t, fillcolor="rgba(30,144,255,0.12)", line_width=0,
                      annotation_text="window", annotation_position="top left")

    seg_changes = series.index[series["segment_index"].diff().fillna(0) != 0]
    for idx in seg_changes:
        if idx == 0:
            continue
        fig.add_vline(x=series.iloc[idx]["display_time"], line_width=1, line_dash="dot",
                      line_color="lightgray")

    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.18),
        xaxis_title="time", yaxis_title="sensor value",
    )
    return fig


def window_zoom_chart(window_df: pd.DataFrame, attn: np.ndarray | None = None) -> go.Figure:
    t = window_df["display_time"]
    fig = go.Figure()
    if attn is not None:
        attn = attn / (attn.max() + 1e-9)
        y_min = float(min(window_df["lcl"].min(), window_df["value"].min()))
        y_max = float(max(window_df["ucl"].max(), window_df["value"].max()))
        for i in range(len(t)):
            fig.add_vrect(
                x0=t.iloc[i] - pd.Timedelta(hours=3),
                x1=t.iloc[i] + pd.Timedelta(hours=3),
                fillcolor=f"rgba(255,0,0,{attn[i] * 0.45:.3f})",
                line_width=0,
            )
        fig.update_yaxes(range=[y_min - 1, y_max + 1])
    fig.add_trace(
        go.Scatter(x=t, y=window_df["value"], mode="lines+markers", name="value",
                   line=dict(color="#1f77b4"))
    )
    fig.add_trace(
        go.Scatter(x=t, y=window_df["target"], mode="lines", name="target",
                   line=dict(color="gray", dash="dash"))
    )
    fig.add_trace(
        go.Scatter(x=t, y=window_df["ucl"], mode="lines", name="UCL",
                   line=dict(color="orange", dash="dot"))
    )
    fig.add_trace(
        go.Scatter(x=t, y=window_df["lcl"], mode="lines",
                   line=dict(color="orange", dash="dot"), showlegend=False)
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def channel_grid_chart(features: np.ndarray) -> go.Figure:
    n_channels = features.shape[1]
    rows, cols = 3, 4
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=CHANNEL_NAMES,
                        vertical_spacing=0.10, horizontal_spacing=0.06)
    for i in range(n_channels):
        r = i // cols + 1
        c = i % cols + 1
        fig.add_trace(
            go.Scatter(y=features[:, i], mode="lines", line=dict(color="#2ca02c"), showlegend=False),
            row=r, col=c,
        )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(showticklabels=False)
    return fig


def prediction_bar(probs: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(CLASSES), y=probs.tolist(),
        marker_color=["#d62728" if i == int(np.argmax(probs)) else "#9ecae1" for i in range(len(CLASSES))],
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=80),
                      yaxis=dict(range=[0, 1]), xaxis_tickangle=-30)
    return fig


def occlusion_bar(importance: dict[str, float]) -> go.Figure:
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#d62728" if v > 0 else "#9ecae1" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="confidence drop when channel masked")
    fig.update_yaxes(autorange="reversed")
    return fig


def comparison_chart(rows: list[dict]) -> go.Figure:
    """rows: each entry has keys: label, prob_v0, prob_v1, class_v0, class_v1."""
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="before", x=df["label"], y=df["prob_v0"], marker_color="#9ecae1"))
    fig.add_trace(go.Bar(name="after", x=df["label"], y=df["prob_v1"], marker_color="#d62728"))
    fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=10, b=60),
                      yaxis=dict(range=[0, 1]), xaxis_tickangle=-25)
    return fig
