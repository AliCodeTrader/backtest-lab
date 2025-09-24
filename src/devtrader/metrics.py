from __future__ import annotations
import pandas as pd

def equity_metrics(equity: pd.Series) -> dict:
    returns = equity.pct_change().dropna()
    if returns.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDrawdown": 0.0, "WinRate": 0.0, "TotalReturn": 0.0}
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252/len(returns)) - 1
    sharpe = (returns.mean() / (returns.std() + 1e-12)) * (252 ** 0.5)
    dd = (equity / equity.cummax()) - 1.0
    max_dd = dd.min()
    win_rate = (returns > 0).mean()
    total_ret = equity.iloc[-1]/equity.iloc[0] - 1
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDrawdown": float(max_dd), "WinRate": float(win_rate), "TotalReturn": float(total_ret)}