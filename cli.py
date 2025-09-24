from __future__ import annotations
import argparse
from pathlib import Path
from src.devtrader.data import load_yf, load_csv
from src.devtrader.strategies import STRATEGIES, strat_ema_rsi, strat_ma_cross, strat_macd
from src.devtrader.backtester import backtest_long_only
from src.devtrader.metrics import equity_metrics
from src.devtrader.plotting import plot_equity_and_drawdown

def main():
    p = argparse.ArgumentParser(description="devtrader-backtest CLI")
    p.add_argument("--ticker", type=str, default="BTC-USD")
    p.add_argument("--start", type=str, default="2022-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--csv", type=str, default=None, help="Optional CSV path")
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--rsi", type=int, default=14)
    p.add_argument("--rsi_min", type=int, default=40)
    p.add_argument("--rsi_max", type=int, default=80)
    p.add_argument("--strategy", type=str, choices=["ema_rsi", "ma_cross", "macd"], default="ema_rsi",
                   help="which strategy to use")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slippage", type=float, default=0.0002)
    p.add_argument("--sl", type=float, default=0.0, help="stop-loss as decimal (e.g., 0.05)")
    p.add_argument("--tp", type=float, default=0.0, help="take-profit as decimal (e.g., 0.10)")
    p.add_argument("--size-mode", type=str, choices=["fixed", "risk"], default="fixed",
                   help="position sizing mode: fixed or risk")
    p.add_argument("--size-fixed", type=float, default=1.0,
                   help="fraction of equity to allocate in fixed mode (0..1)")
    p.add_argument("--risk-frac", type=float, default=0.01,
                   help="risk fraction per trade in risk mode (e.g., 0.01 for 1%)")
    p.add_argument("--atr-len", type=int, default=14, help="ATR length for risk sizing")
    p.add_argument("--atr-mult", type=float, default=2.0, help="ATR multiplier for stop distance")
    p.add_argument("--allow-shorts", dest="allow_shorts", action="store_true", help="enable short selling")
    p.add_argument("--no-shorts", dest="allow_shorts", action="store_false", help="disable short selling")
    p.set_defaults(allow_shorts=True)
    args = p.parse_args()

    df = load_csv(args.csv) if args.csv else load_yf(args.ticker, start=args.start, end=args.end)
    if args.strategy == "ema_rsi":
        sigdf = strat_ema_rsi(
            df,
            fast=args.fast,
            slow=args.slow,
            rsi_len=args.rsi,
            rsi_min=args.rsi_min,
            rsi_max=args.rsi_max,
            allow_shorts=args.allow_shorts,
        )
    elif args.strategy == "ma_cross":
        sigdf = strat_ma_cross(
            df,
            fast=args.fast,
            slow=args.slow,
            allow_shorts=args.allow_shorts,
            ma_type="ema",
        )
    else:  # macd
        sigdf = strat_macd(
            df,
            fast=args.fast,
            slow=args.slow,
            signal=args.rsi,  # reuse --rsi arg as macd signal length
            allow_shorts=args.allow_shorts,
        )
    bt = backtest_long_only(
        sigdf,
        fee=args.fee,
        slippage=args.slippage,
        stop_loss=args.sl,
        take_profit=args.tp,
        size_mode=args.size_mode,
        size_fixed=args.size_fixed,
        risk_frac=args.risk_frac,
        atr_len=args.atr_len,
        atr_mult=args.atr_mult,
        allow_shorts=args.allow_shorts,
    )

    m = equity_metrics(bt["equity"])
    print("\nMetrics:\n", m)

    fig = plot_equity_and_drawdown(bt)
    out = Path("outputs"); out.mkdir(exist_ok=True)
    png = out / f"equity_{(args.csv or args.ticker).replace('/','-').replace('.csv','')}.png"
    fig.savefig(png, dpi=160)
    print(f"Saved plot -> {png}")

if __name__ == "__main__":
    main()