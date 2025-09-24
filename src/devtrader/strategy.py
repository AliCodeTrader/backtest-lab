from __future__ import annotations
import pandas as pd
from .indicators import ema, rsi

def ema_crossover_rsi(df: pd.DataFrame, fast: int = 20, slow: int = 50, rsi_len: int = 14, rsi_min: int = 40, rsi_max: int = 80, allow_shorts: bool = True) -> pd.DataFrame:
    data = df.copy()
    data["ema_fast"] = ema(data["close"], fast)
    data["ema_slow"] = ema(data["close"], slow)
    data["rsi"] = rsi(data["close"], rsi_len)
    data["long"] = (data["ema_fast"] > data["ema_slow"]) & (data["rsi"].between(rsi_min, rsi_max))
    data["short"] = (data["ema_fast"] < data["ema_slow"]) & (data["rsi"].between(100 - rsi_max, 100 - rsi_min))
    if allow_shorts:
        data["signal"] = 0
        data.loc[data["long"], "signal"] = 1
        data.loc[data["short"], "signal"] = -1
    else:
        data["signal"] = data["long"].astype(int)
    return data