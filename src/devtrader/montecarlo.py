

from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
import pandas as pd

__all__ = [
    "iid_bootstrap",
    "block_bootstrap",
    "path_metrics",
    "drawdown",
    "percentiles_table",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _to_1d(arr) -> np.ndarray:
    if isinstance(arr, (pd.Series, pd.Index)):
        return arr.to_numpy().astype(float).ravel()
    if isinstance(arr, pd.DataFrame):
        return arr.squeeze().to_numpy().astype(float).ravel()
    return np.asarray(arr, dtype=float).ravel()


def drawdown(equity: pd.Series | np.ndarray) -> pd.Series:
    """Compute drawdown series from equity path (1D).

    Returns values in [-1, 0].
    """
    eq = _to_1d(equity)
    if eq.size == 0:
        return pd.Series(dtype=float, name="drawdown")
    eq_ser = pd.Series(eq)
    dd = eq_ser / eq_ser.cummax() - 1.0
    dd.name = "drawdown"
    return dd


# -----------------------------------------------------------------------------
# Bootstrap simulators
# -----------------------------------------------------------------------------

def iid_bootstrap(
    returns: pd.Series | np.ndarray,
    n_trials: int = 1000,
    horizon: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """IID bootstrap of returns.

    Parameters
    ----------
    returns : historical strategy returns per bar (1D)
    n_trials : number of simulated paths
    horizon : length of each path (bars). If None, use len(returns).
    seed : RNG seed

    Returns
    -------
    equity_paths : np.ndarray, shape (horizon, n_trials)
        Simulated equity curves starting at 1.0
    """
    r = _to_1d(returns)
    H = int(horizon or len(r))
    rng = np.random.default_rng(seed)
    # Sample indices with replacement per trial
    idx = rng.integers(0, len(r), size=(H, n_trials))
    sampled = r[idx]
    equity = (1.0 + sampled).cumprod(axis=0)
    return equity


def block_bootstrap(
    returns: pd.Series | np.ndarray,
    n_trials: int = 1000,
    horizon: int | None = None,
    block_size: int = 20,
    seed: int | None = None,
) -> np.ndarray:
    """Block bootstrap of returns (stationary bootstrap via fixed blocks).

    We sample contiguous blocks with replacement and concatenate until reaching
    the horizon length.

    Returns equity paths of shape (horizon, n_trials), starting at 1.0.
    """
    r = _to_1d(returns)
    H = int(horizon or len(r))
    B = max(1, int(block_size))
    rng = np.random.default_rng(seed)

    equity = np.empty((H, n_trials), dtype=float)

    for t in range(n_trials):
        out = np.empty(H, dtype=float)
        pos = 0
        while pos < H:
            start = int(rng.integers(0, len(r)))
            end = min(len(r), start + B)
            block = r[start:end]
            take = min(H - pos, block.size)
            out[pos:pos + take] = block[:take]
            pos += take
        equity[:, t] = (1.0 + np.cumprod(1.0 + out) / np.cumprod(np.ones_like(out)))[-H:]  # placeholder to match pattern
        # simpler & correct:
        equity[:, t] = (1.0 + out).cumprod()

    return equity


# -----------------------------------------------------------------------------
# Metrics over simulated paths
# -----------------------------------------------------------------------------

def path_metrics(
    equity_paths: np.ndarray,
    bars_per_year: int = 252,
) -> pd.DataFrame:
    """Compute metrics per simulated path.

    Parameters
    ----------
    equity_paths : array shape (horizon, n_trials)
    bars_per_year : annualization factor for Sharpe/CAGR

    Returns
    -------
    DataFrame with columns: TotalReturn, CAGR, Sharpe, MaxDrawdown
    """
    eq = np.asarray(equity_paths, dtype=float)
    if eq.ndim != 2 or eq.size == 0:
        return pd.DataFrame(columns=["TotalReturn", "CAGR", "Sharpe", "MaxDrawdown"])  # empty

    H, T = eq.shape
    rets = np.zeros_like(eq)
    rets[1:, :] = eq[1:, :] / eq[:-1, :] - 1.0

    total_return = eq[-1, :] / np.clip(eq[0, :], 1e-12, None) - 1.0
    # CAGR ~ (final/initial) ** (bars_per_year / horizon) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        cagr = np.power(np.clip(eq[-1, :] / np.clip(eq[0, :], 1e-12, None), 1e-12, None), bars_per_year / max(1, H-1)) - 1.0
    # Sharpe (annualized) using mean/std of per-bar returns
    mu = np.nanmean(rets[1:, :], axis=0)
    sd = np.nanstd(rets[1:, :], axis=0, ddof=1)
    sharpe = np.where(sd > 0, mu / sd * np.sqrt(bars_per_year), 0.0)

    # Max drawdown per path
    max_dd = np.empty(T, dtype=float)
    for j in range(T):
        series = pd.Series(eq[:, j])
        dd = series / series.cummax() - 1.0
        max_dd[j] = dd.min() if not dd.empty else 0.0

    df = pd.DataFrame({
        "TotalReturn": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
    })
    return df


def percentiles_table(df: pd.DataFrame, percentiles: Iterable[int] = (5, 25, 50, 75, 95)) -> pd.DataFrame:
    """Return a table of percentiles for each metric column in df."""
    if df.empty:
        return pd.DataFrame()
    pct = np.array(list(percentiles), dtype=int)
    out = {}
    for col in df.columns:
        out[col] = np.percentile(df[col].astype(float), pct)
    res = pd.DataFrame(out, index=[f"p{p}" for p in pct])
    return res