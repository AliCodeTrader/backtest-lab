

from __future__ import annotations
from typing import Callable, Dict, Any
import pandas as pd

from .indicators import ema, rsi

__all__ = [
    "strat_ema_rsi",
    "strat_ma_cross",
    "strat_macd",
    "STRATEGIES",
]

# -----------------------------------------------------------------------------
# Strategy functions: each returns a DataFrame with a 'signal' column in {-1,0,+1}
# -----------------------------------------------------------------------------

def strat_ema_rsi(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    rsi_len: int = 14,
    rsi_min: int = 40,
    rsi_max: int = 80,
    allow_shorts: bool = True,
) -> pd.DataFrame:
    data = df.copy()
    data["ema_fast"] = ema(data["close"], fast)
    data["ema_slow"] = ema(data["close"], slow)
    data["rsi"] = rsi(data["close"], rsi_len)

    long = (data["ema_fast"] > data["ema_slow"]) & (data["rsi"].between(rsi_min, rsi_max))
    short = (data["ema_fast"] < data["ema_slow"]) & (data["rsi"].between(100 - rsi_max, 100 - rsi_min))

    if allow_shorts:
        data["signal"] = 0
        data.loc[long, "signal"] = 1
        data.loc[short, "signal"] = -1
    else:
        data["signal"] = long.astype(int)
    return data


def strat_ma_cross(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    allow_shorts: bool = True,
    ma_type: str = "ema",
) -> pd.DataFrame:
    data = df.copy()
    if ma_type.lower() == "ema":
        data["fast"] = ema(data["close"], fast)
        data["slow"] = ema(data["close"], slow)
    else:  # SMA fallback
        data["fast"] = data["close"].rolling(int(fast)).mean()
        data["slow"] = data["close"].rolling(int(slow)).mean()

    long = data["fast"] > data["slow"]
    short = data["fast"] < data["slow"]

    if allow_shorts:
        data["signal"] = 0
        data.loc[long, "signal"] = 1
        data.loc[short, "signal"] = -1
    else:
        data["signal"] = long.astype(int)
    return data


def strat_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    allow_shorts: bool = True,
) -> pd.DataFrame:
    data = df.copy()
    macd_line = ema(data["close"], fast) - ema(data["close"], slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    long = macd_line > signal_line
    short = macd_line < signal_line

    if allow_shorts:
        data["signal"] = 0
        data.loc[long, "signal"] = 1
        data.loc[short, "signal"] = -1
    else:
        data["signal"] = long.astype(int)

    data["macd"] = macd_line
    data["macd_signal"] = signal_line
    return data


# Registry: name -> callable
STRATEGIES: Dict[str, Callable[..., pd.DataFrame]] = {
    "EMA+RSI": strat_ema_rsi,
    "MA Crossover": strat_ma_cross,
    "MACD": strat_macd,
}