from __future__ import annotations
import pandas as pd
import numpy as np
from .indicators import atr as _atr

def backtest_long_only(
    df: pd.DataFrame,
    signal_col: str = "signal",
    fee: float = 0.0005,
    slippage: float = 0.0002,
    stop_loss: float = 0.0,      # e.g., 0.05 for 5% SL
    take_profit: float = 0.0,    # e.g., 0.10 for 10% TP
    allow_shorts: bool = True,   # enable short selling
    # --- sizing params ---
    size_mode: str = "fixed",    # "fixed" or "risk"
    size_fixed: float = 1.0,      # 0..1 fraction of equity (for fixed)
    risk_frac: float = 0.01,      # 1% of equity per trade (for risk)
    atr_len: int = 14,
    atr_mult: float = 2.0,
) -> pd.DataFrame:
    """Backtest with optional SL/TP, position sizing, and short selling.

    Conventions
    -----------
    direction in {-1, 0, +1}; size in [0,1]; exposure = direction*size.
    Costs apply on exposure change (captures enter/exit/flip realistically).
    Entry size is computed at entry and held until exit/flip.
    """
    data = df.copy()

    # ensure arrays are 1-D
    px = data["close"].to_numpy(dtype=float).ravel()
    sig = data[signal_col].astype(int).to_numpy().ravel()

    if not allow_shorts:
        sig = np.maximum(sig, 0)  # clamp to {0,+1}

    n = len(px)
    direction = np.zeros(n, dtype=float)  # -1/0/+1
    size = np.zeros(n, dtype=float)       # 0..1 fraction of equity
    ret = np.zeros(n, dtype=float)
    strat_ret = np.zeros(n, dtype=float)

    # raw returns per bar
    ret[1:] = (px[1:] / px[:-1] - 1.0)

    # precompute ATR if needed for risk sizing
    atr_series = None
    if size_mode == "risk":
        try:
            atr_series = _atr(data, length=int(atr_len)).to_numpy(dtype=float)
        except Exception:
            atr_series = np.zeros(n, dtype=float)

    in_pos = False
    entry_px = np.nan
    entry_size = 0.0
    entry_dir = 0.0  # -1 or +1

    for i in range(1, n):
        desired_dir = int(sig[i-1])  # act on prior bar's signal

        # If in position, check SL/TP relative to entry depending on direction
        if in_pos:
            if entry_dir > 0:  # long
                cur_cum = px[i] / entry_px - 1.0
            else:              # short
                cur_cum = entry_px / px[i] - 1.0
            if stop_loss > 0 and cur_cum <= -abs(stop_loss):
                desired_dir = 0
            elif take_profit > 0 and cur_cum >= abs(take_profit):
                desired_dir = 0

        prev_dir = direction[i-1]

        if prev_dir == 0 and desired_dir != 0:
            # ENTER new position at bar i, compute size
            entry_px = px[i]
            in_pos = True
            entry_dir = float(np.sign(desired_dir))
            if size_mode == "fixed":
                entry_size = float(np.clip(size_fixed, 0.0, 1.0))
            elif size_mode == "risk":
                atr_val = atr_series[i] if atr_series is not None else 0.0
                stop_dist = max(1e-12, (atr_mult * atr_val) / max(1e-12, entry_px))
                entry_size = float(np.clip(risk_frac / stop_dist, 0.0, 1.0))
            else:
                entry_size = 1.0
            direction[i] = entry_dir
            size[i] = entry_size

        elif prev_dir != 0 and desired_dir == 0:
            # EXIT position
            in_pos = False
            entry_px = np.nan
            entry_size = 0.0
            entry_dir = 0.0
            direction[i] = 0.0
            size[i] = 0.0

        elif prev_dir != 0 and desired_dir != 0 and np.sign(prev_dir) != np.sign(desired_dir):
            # FLIP direction: exit then enter opposite on same bar
            entry_px = px[i]
            in_pos = True
            entry_dir = float(np.sign(desired_dir))
            if size_mode == "fixed":
                entry_size = float(np.clip(size_fixed, 0.0, 1.0))
            elif size_mode == "risk":
                atr_val = atr_series[i] if atr_series is not None else 0.0
                stop_dist = max(1e-12, (atr_mult * atr_val) / max(1e-12, entry_px))
                entry_size = float(np.clip(risk_frac / stop_dist, 0.0, 1.0))
            else:
                entry_size = 1.0
            direction[i] = entry_dir
            size[i] = entry_size

        else:
            # HOLD (keep previous direction & size)
            direction[i] = prev_dir
            size[i] = entry_size if in_pos else 0.0

        # strategy return over (i-1 -> i) uses prior exposure
        exposure_prev = direction[i-1] * (size[i-1] if i-1 >= 0 else 0.0)
        strat_ret[i] = exposure_prev * ret[i]

    # transaction costs based on exposure change (captures enter/exit/flip correctly)
    exposure = direction * size
    exposure_changes = np.abs(np.diff(exposure, prepend=0.0))
    tc = exposure_changes * (fee + slippage)
    strat_ret -= tc

    equity = (1.0 + strat_ret).cumprod()

    out = data.copy()
    out["direction"] = direction
    out["position"] = (direction != 0).astype(float)  # backward-compat (0/1)
    out["size"] = size
    out["ret"] = ret
    out["strategy_ret"] = strat_ret
    out["equity"] = equity
    return out