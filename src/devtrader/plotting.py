from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go

def plot_equity_and_drawdown(bt: pd.DataFrame):
    eq = bt["equity"]
    dd = (eq / eq.cummax()) - 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", name="Equity"))
    fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name="Drawdown", fill="tozeroy", opacity=0.3, yaxis="y2"))

    # Setup dual y-axes
    fig.update_layout(
        title="Equity & Drawdown",
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Equity"),
        yaxis2=dict(title="Drawdown", overlaying="y", side="right"),
        legend=dict(x=0, y=1.1, orientation="h"),
    )
    return fig