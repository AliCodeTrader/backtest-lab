from __future__ import annotations
import pandas as pd
import yfinance as yf

def load_yf(ticker: str, start: str | None = None, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} with given parameters")
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df = df[["open","high","low","close","volume"]]
    df.index.name = "date"
    return df

def load_csv(path: str, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
    cols = {c.lower(): c for c in df.columns}
    rename_map = {cols.get("open","open"):"open", cols.get("high","high"):"high", cols.get("low","low"):"low",
                  cols.get("close","close"):"close", cols.get("volume","volume"):"volume"}
    df = df.rename(columns=rename_map)[["open","high","low","close","volume"]]
    return df