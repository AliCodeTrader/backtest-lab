from __future__ import annotations
import itertools
from typing import Iterable, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from .strategy import ema_crossover_rsi
from .backtester import backtest_long_only
from .metrics import equity_metrics


def _grid_search_train(
    df: pd.DataFrame,
    fast_range: Iterable[int],
    slow_range: Iterable[int],
    rsi_len_range: Iterable[int],
    rsi_min: int,
    rsi_max: int,
    fee: float,
    slippage: float,
    stop_loss: float,
    take_profit: float,
    allow_shorts: bool,
    size_mode: str,
    size_fixed: float,
    risk_frac: float,
    atr_len: int,
    atr_mult: float,
    optimize_by: str = "TotalReturn",
) -> Dict[str, Any]:
    """Run a small grid on TRAIN slice and return best params (dict).
    optimize_by ∈ {"TotalReturn", "Sharpe", "CAGR"}
    """
    best_row: Dict[str, Any] | None = None
    best_score = -np.inf

    for fast, slow, rsi_len in itertools.product(fast_range, slow_range, rsi_len_range):
        if fast >= slow:
            continue
        sig = ema_crossover_rsi(
            df,
            fast=fast,
            slow=slow,
            rsi_len=rsi_len,
            rsi_min=rsi_min,
            rsi_max=rsi_max,
            allow_shorts=allow_shorts,
        )
        bt = backtest_long_only(
            sig,
            fee=fee,
            slippage=slippage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            allow_shorts=allow_shorts,
            size_mode=size_mode,
            size_fixed=size_fixed,
            risk_frac=risk_frac,
            atr_len=atr_len,
            atr_mult=atr_mult,
        )
        m = equity_metrics(bt["equity"])  # dict
        score = float(m.get(optimize_by, float("-inf")))
        if np.isnan(score):
            score = -np.inf
        if score > best_score:
            best_score = score
            best_row = {
                "fast": int(fast),
                "slow": int(slow),
                "rsi_len": int(rsi_len),
                "score": float(score),
                "metrics": m,
            }

    if best_row is None:
        # fallback: basic defaults
        best_row = {
            "fast": 20,
            "slow": 50,
            "rsi_len": 14,
            "score": float("nan"),
            "metrics": {},
        }
    return best_row


def walk_forward(
    df: pd.DataFrame,
    # parameter search ranges
    fast_range: Iterable[int],
    slow_range: Iterable[int],
    rsi_len_range: Iterable[int],
    # strategy/indicator params
    rsi_min: int,
    rsi_max: int,
    # trading frictions
    fee: float,
    slippage: float,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    allow_shorts: bool = True,
    # sizing
    size_mode: str = "fixed",
    size_fixed: float = 1.0,
    risk_frac: float = 0.01,
    atr_len: int = 14,
    atr_mult: float = 2.0,
    # walk-forward config
    train_bars: int = 500,
    test_bars: int = 100,
    max_folds: int = 5,
    split_mode: str = "rolling",  # "rolling" or "expanding"
    optimize_by: str = "TotalReturn",
) -> Tuple[pd.DataFrame, pd.Series, List[Dict[str, Any]]]:
    """
    Perform walk-forward optimization and evaluation.

    Returns
    -------
    folds_df : pd.DataFrame
        Per-fold results with train/test dates, chosen params, and OOS metrics.
    oos_equity : pd.Series
        Concatenated out-of-sample equity (starts at 1.0 at the first OOS bar).
    best_params_per_fold : list[dict]
        List of best parameter dicts for each fold.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        # Ensure we can slice cleanly by index order
        df = df.copy().sort_index()
    else:
        df = df.sort_index()

    n = len(df)
    if n < train_bars + test_bars:
        raise ValueError("Not enough data for the requested train/test sizes.")

    folds: List[Dict[str, Any]] = []
    ret_segments: List[pd.Series] = []
    best_params_list: List[Dict[str, Any]] = []

    start = 0
    train_start = 0
    fold_count = 0

    while True:
        train_end = start + train_bars
        test_end = train_end + test_bars
        if test_end > n or fold_count >= max_folds:
            break

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end]

        # --- optimize on TRAIN ---
        best = _grid_search_train(
            train_df,
            fast_range=fast_range,
            slow_range=slow_range,
            rsi_len_range=rsi_len_range,
            rsi_min=rsi_min,
            rsi_max=rsi_max,
            fee=fee,
            slippage=slippage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            allow_shorts=allow_shorts,
            size_mode=size_mode,
            size_fixed=size_fixed,
            risk_frac=risk_frac,
            atr_len=atr_len,
            atr_mult=atr_mult,
            optimize_by=optimize_by,
        )
        best_params_list.append(best)

        # --- evaluate on TEST (OOS) ---
        sig_test = ema_crossover_rsi(
            test_df,
            fast=best["fast"],
            slow=best["slow"],
            rsi_len=best["rsi_len"],
            rsi_min=rsi_min,
            rsi_max=rsi_max,
            allow_shorts=allow_shorts,
        )
        bt_test = backtest_long_only(
            sig_test,
            fee=fee,
            slippage=slippage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            allow_shorts=allow_shorts,
            size_mode=size_mode,
            size_fixed=size_fixed,
            risk_frac=risk_frac,
            atr_len=atr_len,
            atr_mult=atr_mult,
        )

        # Collect OOS metrics
        m_oos = equity_metrics(bt_test["equity"])  # equity starts at 1 in this slice

        # For global OOS equity, concatenate strategy returns and compound
        seg_ret = bt_test["strategy_ret"].copy()
        seg_ret.name = "strategy_ret"
        ret_segments.append(seg_ret)

        folds.append({
            "fold": fold_count + 1,
            "train_start": train_df.index[0],
            "train_end": train_df.index[-1],
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            "fast": best["fast"],
            "slow": best["slow"],
            "rsi_len": best["rsi_len"],
            "opt_metric": optimize_by,
            "opt_score": best.get("score", np.nan),
            **{f"OOS_{k}": v for k, v in m_oos.items()},
        })

        # advance window
        if split_mode == "expanding":
            start += test_bars
            # expanding: keep train_start=0; train_end grows by +test_bars next loop
        else:  # rolling
            start += test_bars
            train_start += test_bars

        fold_count += 1

    folds_df = pd.DataFrame(folds)

    # Build concatenated OOS equity from returns
    if ret_segments:
        all_ret = pd.concat(ret_segments)
        oos_equity = (1.0 + all_ret.fillna(0.0)).cumprod()
        oos_equity.name = "equity"
    else:
        oos_equity = pd.Series(dtype=float, name="equity")

    return folds_df, oos_equity, best_params_list
