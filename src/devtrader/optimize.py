from __future__ import annotations
import itertools
import pandas as pd
from typing import Iterable, Dict, Any
from .strategy import ema_crossover_rsi
from .backtester import backtest_long_only
from .metrics import equity_metrics

def grid_search(
    df: pd.DataFrame,
    fast_range: Iterable[int],
    slow_range: Iterable[int],
    rsi_len_range: Iterable[int],
    rsi_min: int,
    rsi_max: int,
    fee: float,
    slippage: float,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    ساده و سریع: sweep روی fast/slow/rsi_len (با شرط fast < slow).
    rsi_min/max و sl/tp ثابت می‌مونند.
    خروجی: جدول نتایج مرتب‌شده بر اساس TotalReturn.
    """
    rows: list[Dict[str, Any]] = []

    for fast, slow, rsi_len in itertools.product(fast_range, slow_range, rsi_len_range):
        if fast >= slow:
            continue  # کراس‌اوور منطقی
        sig = ema_crossover_rsi(df, fast=fast, slow=slow, rsi_len=rsi_len, rsi_min=rsi_min, rsi_max=rsi_max)
        bt = backtest_long_only(
            sig,
            fee=fee,
            slippage=slippage,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        m = equity_metrics(bt["equity"])
        rows.append({
            "fast": fast,
            "slow": slow,
            "rsi_len": rsi_len,
            **m,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res = res.sort_values("TotalReturn", ascending=False).reset_index(drop=True)
    return res.head(top_n)