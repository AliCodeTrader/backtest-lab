import streamlit as st
from src.devtrader.data import load_yf, load_csv
from src.devtrader.strategy import ema_crossover_rsi
from src.devtrader.backtester import backtest_long_only
from src.devtrader.metrics import equity_metrics
from src.devtrader.plotting import plot_equity_and_drawdown
from src.devtrader.walkforward import walk_forward

from src.devtrader.strategies import STRATEGIES, strat_ema_rsi, strat_ma_cross, strat_macd

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Iterable, Dict, Any

from src.devtrader.montecarlo import iid_bootstrap, block_bootstrap, path_metrics, percentiles_table

# --- Extra imports for report export ---
import io, base64, json, datetime

# --- Local grid search helper (no extra file needed) ---
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
    allow_shorts: bool = True,
    top_n: int = 20,
) -> pd.DataFrame:
    import itertools
    from src.devtrader.strategy import ema_crossover_rsi
    from src.devtrader.backtester import backtest_long_only
    from src.devtrader.metrics import equity_metrics

    rows: list[Dict[str, Any]] = []
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            for rsi_len in rsi_len_range:
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

st.set_page_config(page_title="devtrader-backtest", layout="wide")
st.title("devtrader-backtest 🧪📈")

strategy_name = st.selectbox("Strategy", ["EMA+RSI", "MA Crossover", "MACD"], index=0)

src_choice = st.radio("Data source", ["yfinance","CSV"], horizontal=True)

if src_choice == "yfinance":
    col1, col2, col3 = st.columns(3)
    with col1: ticker = st.text_input("Ticker", "BTC-USD")
    with col2: start = st.text_input("Start", "2022-01-01")
    with col3: end = st.text_input("End (optional)", "") or None
    try:
        df = load_yf(ticker, start=start, end=end)
    except Exception as e:
        st.error(f"Failed to load data from yfinance: {e}")
        st.stop()
else:
    # Safer CSV handling: allow upload OR a path; validate schema
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    csv_path = st.text_input("CSV path (optional)", "")

    if uploaded is not None:
        import pandas as pd
        try:
            df = pd.read_csv(uploaded, parse_dates=["date"])  # expects a 'date' column
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            st.stop()
        # normalize columns like load_csv does
        cols = {c.lower(): c for c in df.columns}
        if "date" not in [c.lower() for c in df.columns]:
            st.error("CSV must include a 'date' column.")
            st.stop()
        rename_map = {
            cols.get("open", "open"): "open",
            cols.get("high", "high"): "high",
            cols.get("low", "low"): "low",
            cols.get("close", "close"): "close",
            cols.get("volume", "volume"): "volume",
        }
        df = df.rename(columns=rename_map)
        df["date"] = pd.to_datetime(df["date"])  # ensure datetime
        df = df.set_index("date").sort_index()[["open", "high", "low", "close", "volume"]]
    elif csv_path:
        try:
            df = load_csv(csv_path)
        except FileNotFoundError:
            st.error(f"CSV not found at: {csv_path}")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            st.stop()
    else:
        st.info("No CSV provided. Upload a file above or switch to yfinance.")
        st.stop()

tab_params, tab_metrics, tab_grid, tab_compare, tab_reports = st.tabs(["Parameters", "Metrics", "Grid Search", "Compare Runs", "Reports"])

with tab_params:
    st.subheader("Parameters")
    if strategy_name == "EMA+RSI":
        colA, colB, colC, colD, colE = st.columns(5)
        with colA: fast = st.number_input("EMA Fast", 5, 200, 20, key="fast_param")
        with colB: slow = st.number_input("EMA Slow", 10, 400, 50, key="slow_param")
        with colC: rsi_len = st.number_input("RSI Length", 2, 50, 14, key="rsi_param")
        with colD: rsi_min = st.number_input("RSI Min", 0, 100, 40)
        with colE: rsi_max = st.number_input("RSI Max", 0, 100, 80)
        params_extra = {"rsi_len": int(rsi_len), "rsi_min": int(rsi_min), "rsi_max": int(rsi_max)}
    elif strategy_name == "MA Crossover":
        colA, colB, colC = st.columns(3)
        with colA: fast = st.number_input("MA Fast", 5, 200, 20, key="fast_param")
        with colB: slow = st.number_input("MA Slow", 10, 400, 50, key="slow_param")
        with colC: ma_type = st.radio("MA Type", ["EMA", "SMA"], horizontal=True)
        # defaults for hidden RSI fields
        rsi_len, rsi_min, rsi_max = 14, 40, 80
        params_extra = {"ma_type": ma_type}
    else:  # MACD
        colA, colB, colC = st.columns(3)
        with colA: fast = st.number_input("MACD Fast", 2, 100, 12, key="fast_param")
        with colB: slow = st.number_input("MACD Slow", 5, 200, 26, key="slow_param")
        with colC: macd_signal = st.number_input("Signal Length", 2, 50, 9, key="macd_signal_param")
        # defaults for hidden RSI fields
        rsi_len, rsi_min, rsi_max = 14, 40, 80
        params_extra = {"macd_signal": int(macd_signal)}

    allow_shorts = st.checkbox("Allow Shorts", value=True)

    colF, colG = st.columns(2)
    with colF: fee = st.number_input("Fee", 0.0, 0.01, 0.0005, step=0.0001, format="%.4f")
    with colG: slip = st.number_input("Slippage", 0.0, 0.01, 0.0002, step=0.0001, format="%.4f")

    # Add Stop Loss and Take Profit inputs
    colH, colI = st.columns(2)
    with colH:
        sl = st.number_input("Stop Loss (decimal)", 0.0, 1.0, 0.0, step=0.01, format="%.2f")
    with colI:
        tp = st.number_input("Take Profit (decimal)", 0.0, 1.0, 0.0, step=0.01, format="%.2f")

    # Position Sizing
    st.subheader("Position Sizing")
    size_mode = st.radio("Sizing mode", ["fixed", "risk"], horizontal=True)
    if size_mode == "fixed":
        size_fixed = st.number_input("Fixed Size (0..1)", 0.0, 1.0, 1.0, step=0.05)
        risk_frac = 0.01
        atr_len = 14
        atr_mult = 2.0
    else:
        size_fixed = 0.0
        risk_frac = st.number_input("Risk Fraction per trade", 0.0, 1.0, 0.01, step=0.005, format="%.3f")
        atr_len = st.number_input("ATR Length", 2, 200, 14)
        atr_mult = st.number_input("ATR Multiplier", 0.1, 10.0, 2.0, step=0.1)

with tab_grid:
    if strategy_name != "EMA+RSI":
        st.info("Grid Search is currently available for EMA+RSI strategy only.")
    st.write("Sweep روی پارامترها برای یافتن ترکیب‌های بهتر. شرط: fast < slow.")
    c1, c2, c3 = st.columns(3)
    with c1:
        fast_start = st.number_input("fast start", 5, 200, 10)
        fast_end   = st.number_input("fast end",   6, 400, 40)
        fast_step  = st.number_input("fast step",  1, 100, 5)
    with c2:
        slow_start = st.number_input("slow start", 10, 400, 30)
        slow_end   = st.number_input("slow end",   11, 800, 120)
        slow_step  = st.number_input("slow step",  1, 100, 5)
    with c3:
        rsi_len_start = st.number_input("RSI len start", 2, 50, 10)
        rsi_len_end   = st.number_input("RSI len end",   3, 60, 20)
        rsi_len_step  = st.number_input("RSI len step",  1, 20, 2)

    run = st.button("Run Grid Search")
    if run:
        fast_range = range(int(fast_start), int(fast_end) + 1, int(fast_step))
        slow_range = range(int(slow_start), int(slow_end) + 1, int(slow_step))
        rsi_len_range = range(int(rsi_len_start), int(rsi_len_end) + 1, int(rsi_len_step))

        resdf = grid_search(
            df,
            fast_range=fast_range,
            slow_range=slow_range,
            rsi_len_range=rsi_len_range,
            rsi_min=int(rsi_min),
            rsi_max=int(rsi_max),
            fee=float(fee),
            slippage=float(slip),
            stop_loss=float(sl),
            take_profit=float(tp),
            allow_shorts=allow_shorts,
            top_n=50,
        )

        if resdf.empty:
            st.warning("نتیجه‌ای پیدا نشد. بازه‌ها را چک کن.")
        else:
            st.subheader("Top Results")
            st.dataframe(resdf)

            # Load best params button
            best_row = resdf.iloc[0]
            st.write("Best Params:", dict(best_row))
            if st.button("Load Best Params"):
                st.session_state["fast_param"] = int(best_row["fast"])
                st.session_state["slow_param"] = int(best_row["slow"])
                st.session_state["rsi_param"] = int(best_row["rsi_len"])
                st.success("Best parameters loaded into form above!")

            # Heatmap ساده برای fast vs slow روی بهترین rsi_len
            best_rsi = int(resdf.iloc[0]["rsi_len"]) if "rsi_len" in resdf.columns else int(rsi_len)
            hm_df = resdf[resdf["rsi_len"] == best_rsi].pivot_table(
                index="fast", columns="slow", values="TotalReturn", aggfunc="max"
            )
            if not hm_df.empty:
                fig_hm, ax = plt.subplots(figsize=(6, 4))
                im = ax.imshow(hm_df.values, aspect="auto", origin="lower")
                ax.set_xticks(range(len(hm_df.columns)))
                ax.set_yticks(range(len(hm_df.index)))
                ax.set_xticklabels(hm_df.columns)
                ax.set_yticklabels(hm_df.index)
                ax.set_xlabel("slow")
                ax.set_ylabel("fast")
                ax.set_title(f"TotalReturn Heatmap (rsi_len={best_rsi})")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig_hm)
            else:
                st.info("داده کافی برای رسم Heatmap وجود ندارد.")

with tab_metrics:
    with st.expander("🧭 Walk-Forward Analysis"):
        if strategy_name != "EMA+RSI":
            st.info("Walk-Forward is currently available for EMA+RSI strategy only.")
        st.write("Optimize on TRAIN, evaluate on TEST (out-of-sample). Helps avoid overfitting.")

    c1, c2, c3 = st.columns(3)
    with c1:
        split_mode = st.radio("Split mode", ["rolling", "expanding"], horizontal=True)
        optimize_by = st.selectbox("Optimize by", ["TotalReturn", "Sharpe", "CAGR"], index=0)
    with c2:
        train_bars = st.number_input("Train bars", 50, 5000, 500, step=50)
        test_bars  = st.number_input("Test bars", 20, 2000, 100, step=20)
    with c3:
        max_folds  = st.number_input("Max folds", 1, 50, 5, step=1)

    st.markdown("**Parameter ranges (TRAIN grid search)**")
    g1, g2, g3 = st.columns(3)
    with g1:
        wf_fast_start = st.number_input("fast start (WF)", 5, 200, 10)
        wf_fast_end   = st.number_input("fast end (WF)",   6, 400, 40)
        wf_fast_step  = st.number_input("fast step (WF)",  1, 100, 5)
    with g2:
        wf_slow_start = st.number_input("slow start (WF)", 10, 400, 30)
        wf_slow_end   = st.number_input("slow end (WF)",   11, 800, 120)
        wf_slow_step  = st.number_input("slow step (WF)",  1, 100, 5)
    with g3:
        wf_rsi_len_start = st.number_input("RSI len start (WF)", 2, 50, 10)
        wf_rsi_len_end   = st.number_input("RSI len end (WF)",   3, 60, 20)
        wf_rsi_len_step  = st.number_input("RSI len step (WF)",  1, 20, 2)

    run_wf = st.button("Run Walk-Forward")
    if run_wf:
        fast_range = range(int(wf_fast_start), int(wf_fast_end) + 1, int(wf_fast_step))
        slow_range = range(int(wf_slow_start), int(wf_slow_end) + 1, int(wf_slow_step))
        rsi_len_range = range(int(wf_rsi_len_start), int(wf_rsi_len_end) + 1, int(wf_rsi_len_step))

        try:
            folds_df, oos_equity, best_params = walk_forward(
                df,
                fast_range=fast_range,
                slow_range=slow_range,
                rsi_len_range=rsi_len_range,
                rsi_min=int(rsi_min),
                rsi_max=int(rsi_max),
                fee=float(fee),
                slippage=float(slip),
                stop_loss=float(sl),
                take_profit=float(tp),
                allow_shorts=bool(allow_shorts),
                size_mode=size_mode,
                size_fixed=float(size_fixed),
                risk_frac=float(risk_frac),
                atr_len=int(atr_len),
                atr_mult=float(atr_mult),
                train_bars=int(train_bars),
                test_bars=int(test_bars),
                max_folds=int(max_folds),
                split_mode=split_mode,
                optimize_by=optimize_by,
            )
        except Exception as e:
            st.error(f"Walk-Forward failed: {e}")
        else:
            if not folds_df.empty:
                st.subheader("Per-Fold Results (OOS)")
                st.dataframe(folds_df)
                # Download folds CSV
                csv_folds = folds_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Folds CSV",
                    data=csv_folds,
                    file_name="walkforward_folds.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No folds produced; check data length or parameters.")

            # OOS Equity
            if not oos_equity.empty:
                st.subheader("OOS Equity Curve")
                fig_oos, ax = plt.subplots(figsize=(10, 4))
                ax.plot(oos_equity.index, oos_equity.values)
                ax.set_title("Walk-Forward Out-of-Sample Equity")
                ax.set_xlabel("date")
                ax.set_ylabel("equity")
                st.pyplot(fig_oos)

                # OOS metrics
                oos_m = equity_metrics(oos_equity)
                st.markdown("**OOS Metrics**")
                st.json(oos_m)

                # Download OOS equity CSV
                csv_oos = oos_equity.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download OOS Equity CSV",
                    data=csv_oos,
                    file_name="walkforward_oos_equity.csv",
                    mime="text/csv",
                )
            else:
                st.info("OOS equity empty.")

if strategy_name == "EMA+RSI":
    sigdf = strat_ema_rsi(
        df,
        fast=int(fast), slow=int(slow),
        rsi_len=int(rsi_len), rsi_min=int(rsi_min), rsi_max=int(rsi_max),
        allow_shorts=allow_shorts,
    )
elif strategy_name == "MA Crossover":
    sigdf = strat_ma_cross(
        df,
        fast=int(fast), slow=int(slow),
        allow_shorts=allow_shorts,
        ma_type=params_extra.get("ma_type", "EMA").lower(),
    )
else:  # MACD
    sigdf = strat_macd(
        df,
        fast=int(fast), slow=int(slow),
        signal=int(params_extra.get("macd_signal", 9)),
        allow_shorts=allow_shorts,
    )
res = backtest_long_only(
    sigdf,
    fee=fee,
    slippage=slip,
    stop_loss=sl,
    take_profit=tp,
    size_mode=size_mode,
    size_fixed=float(size_fixed),
    risk_frac=float(risk_frac),
    atr_len=int(atr_len),
    atr_mult=float(atr_mult),
    allow_shorts=allow_shorts,
)

with tab_metrics:
    st.subheader("Metrics")
    metrics = equity_metrics(res["equity"])
    st.json(metrics)

    st.subheader("Equity (interactive)")
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=res.index, y=res["equity"], mode="lines", name="Equity"))
    fig_equity.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_equity, use_container_width=True)

    # also create a matplotlib figure for report export (not shown)
    fig_report = plt.figure(figsize=(8, 3))
    axr = fig_report.add_subplot(111)
    axr.plot(res.index, res["equity"], label="Equity")
    axr.set_title("Equity")
    axr.set_xlabel("date")
    axr.set_ylabel("equity")
    axr.legend(loc="best")

with tab_reports:
    st.write("Create a self-contained HTML report with parameters, metrics, and the equity chart.")

    # Prepare report data
    source_info = {"source": src_choice}
    try:
        if src_choice == "yfinance":
            source_info.update({"ticker": ticker, "start": start, "end": end})
        else:
            source_info.update({"csv_path": csv_path})
    except Exception:
        pass

    sizing_info = {
        "size_mode": size_mode,
        "size_fixed": float(size_fixed),
        "risk_frac": float(risk_frac),
        "atr_len": int(atr_len),
        "atr_mult": float(atr_mult),
    }

    strat_info = {
        "strategy": strategy_name,
        "fast": int(fast),
        "slow": int(slow),
        "rsi_len": int(rsi_len),
        "rsi_min": int(rsi_min),
        "rsi_max": int(rsi_max),
        "allow_shorts": bool(allow_shorts),
    }
    if strategy_name == "MA Crossover":
        strat_info["ma_type"] = params_extra.get("ma_type", "EMA")
    if strategy_name == "MACD":
        strat_info["macd_signal"] = int(params_extra.get("macd_signal", 9))

    run_params = {
        **source_info,
        **strat_info,
        "fee": float(fee),
        "slippage": float(slip),
        "stop_loss": float(sl),
        "take_profit": float(tp),
        **sizing_info,
    }

    # Encode equity chart as base64 PNG
    img_b64 = None
    try:
        _buf = io.BytesIO()
        fig_report.savefig(_buf, format="png", bbox_inches="tight")
        img_b64 = base64.b64encode(_buf.getvalue()).decode()
    except Exception:
        img_b64 = None

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        "<title>devtrader-backtest — Report</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;margin:24px} .card{border:1px solid #eee;border-radius:12px;padding:16px;margin-bottom:16px} h1,h2{margin:4px 0 8px 0} pre{background:#f7f7f7;padding:10px;border-radius:8px;overflow:auto} table{border-collapse: collapse} td,th{border:1px solid #ddd;padding:6px 10px}</style>",
        "</head><body>",
        f"<h1>devtrader-backtest — Run Report</h1><div>Generated: {ts}</div>",
        "<div class='card'><h2>Parameters</h2><pre>",
        json.dumps(run_params, indent=2),
        "</pre></div>",
        "<div class='card'><h2>Metrics</h2><pre>",
        json.dumps(metrics, indent=2),
        "</pre></div>",
    ]
    if img_b64:
        html_parts.append("<div class='card'><h2>Equity Chart</h2>")
        html_parts.append(f"<img src='data:image/png;base64,{img_b64}' alt='Equity chart' />")
        html_parts.append("</div>")
    html_parts.append("</body></html>")

    report_html = "".join(html_parts)

    default_fname = f"report_{(ticker if src_choice=='yfinance' else 'CSV')}_{strategy_name.replace(' ','')}.html"
    st.download_button(
        label="⬇️ Download HTML Report",
        data=report_html.encode("utf-8"),
        file_name=default_fname,
        mime="text/html",
    )

with tab_metrics:
    with st.expander("🎲 Monte Carlo (risk simulation)"):
        st.write("Simulate many alternative equity paths by bootstrapping strategy returns to estimate risk (drawdowns, tails).")

    # Inputs
    mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
    with mc_col1:
        mc_trials = st.number_input("Trials", 100, 20000, 1000, step=100)
    with mc_col2:
        horizon_default = int(len(res))
        mc_horizon = st.number_input("Horizon (bars)", 10, max(horizon_default, 10), horizon_default, step=10)
    with mc_col3:
        mc_method = st.radio("Method", ["iid", "block"], horizontal=True)
    with mc_col4:
        mc_seed = st.number_input("Seed (optional)", 0, 10_000_000, 0, step=1)

    block_size = None
    if mc_method == "block":
        block_size = st.number_input("Block size", 2, 500, 20, step=1)

    alpha = st.number_input("VaR alpha (%)", 1, 25, 5, step=1)

    run_mc = st.button("Run Monte Carlo")
    if run_mc:
        try:
            rets = res["strategy_ret"].dropna().to_numpy().astype(float)
            if rets.size == 0:
                st.warning("No strategy returns available.")
                st.stop()
            H = int(mc_horizon)
            seed = int(mc_seed) if mc_seed != 0 else None

            if mc_method == "iid":
                paths = iid_bootstrap(rets, n_trials=int(mc_trials), horizon=H, seed=seed)
            else:
                paths = block_bootstrap(rets, n_trials=int(mc_trials), horizon=H, block_size=int(block_size or 20), seed=seed)

            mdf = path_metrics(paths)
        except Exception as e:
            st.error(f"Monte Carlo failed: {e}")
        else:
            st.subheader("Monte Carlo Results")
            # Percentiles table
            pct_tbl = percentiles_table(mdf, percentiles=(1,5,10,25,50,75,90,95,99))
            st.markdown("**Percentiles (across trials)**")
            st.dataframe(pct_tbl)

            # VaR / CVaR on TotalReturn
            try:
                import numpy as np
                tr = mdf["TotalReturn"].astype(float).to_numpy()
                var_level = float(np.percentile(tr, alpha))
                cvar = float(tr[tr <= var_level].mean()) if (tr <= var_level).any() else float(var_level)
                st.markdown(f"**VaR{alpha}% (TotalReturn):** {var_level:.3f}  |  **CVaR{alpha}%:** {cvar:.3f}")
            except Exception:
                pass

            # Histograms (Plotly)
            fig_tr = go.Figure()
            fig_tr.add_histogram(x=mdf["TotalReturn"].astype(float), nbinsx=50, name="TotalReturn")
            fig_tr.update_layout(title="Histogram: TotalReturn", bargap=0.02, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_tr, use_container_width=True)

            fig_dd = go.Figure()
            fig_dd.add_histogram(x=mdf["MaxDrawdown"].astype(float), nbinsx=50, name="MaxDrawdown")
            fig_dd.update_layout(title="Histogram: MaxDrawdown", bargap=0.02, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_dd, use_container_width=True)

            # Download metrics CSV
            csv_mc = mdf.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Monte Carlo Metrics CSV",
                data=csv_mc,
                file_name="montecarlo_metrics.csv",
                mime="text/csv",
            )

with tab_compare:
    # Prepare a default run name based on current params (with sizing)
    size_tag = (
        f"sz:fixed-{float(size_fixed):.2f}" if size_mode == "fixed"
        else f"sz:risk-{float(risk_frac):.3f}-atr{int(atr_len)}x{float(atr_mult):.1f}"
    )
    default_name = (
        f"{(ticker if src_choice=='yfinance' else (csv_path or 'CSV'))}"
        f"-{strategy_name.replace(' ', '')}"
        f"-f{int(fast)}-s{int(slow)}"
        f"-r{int(rsi_len)}-sl{sl:.2f}-tp{tp:.2f}-{size_tag}-shorts:{'on' if allow_shorts else 'off'}"
    )
    run_name = st.text_input("Run name", default_name)

    # Initialize storage
    if "runs" not in st.session_state:
        st.session_state["runs"] = []

    col_save, col_clear = st.columns([3,1])
    with col_save:
        if st.button("Save current run"):
            # store params, metrics, and equity series
            run = {
                "name": run_name,
                "params": {
                    "fast": int(fast),
                    "slow": int(slow),
                    "rsi_len": int(rsi_len),
                    "rsi_min": int(rsi_min),
                    "rsi_max": int(rsi_max),
                    "strategy": strategy_name,
                    **({"ma_type": params_extra.get("ma_type")} if strategy_name == "MA Crossover" else {}),
                    **({"macd_signal": params_extra.get("macd_signal")} if strategy_name == "MACD" else {}),
                    "fee": float(fee),
                    "slippage": float(slip),
                    "stop_loss": float(sl),
                    "take_profit": float(tp),
                    "size_mode": size_mode,
                    "size_fixed": float(size_fixed),
                    "risk_frac": float(risk_frac),
                    "atr_len": int(atr_len),
                    "atr_mult": float(atr_mult),
                    "allow_shorts": bool(allow_shorts),
                    "source": src_choice,
                    "ticker": ticker if src_choice=="yfinance" else (csv_path or "CSV"),
                },
                "metrics": metrics,
                "equity": res[["equity"]].copy(),
            }
            st.session_state["runs"].append(run)
            st.success(f"Saved run: {run_name}")
    with col_clear:
        if st.button("Clear all runs"):
            st.session_state["runs"] = []
            st.warning("All saved runs cleared.")

    runs = st.session_state.get("runs", [])
    st.caption(f"Saved runs: {len(runs)}")

    combined = None
    if runs:
        # Show a small table of saved runs + top metrics
        preview_rows = []
        for r in runs:
            preview_rows.append({
                "name": r["name"],
                "TotalReturn": r["metrics"].get("TotalReturn", 0.0),
                "CAGR": r["metrics"].get("CAGR", 0.0),
                "Sharpe": r["metrics"].get("Sharpe", 0.0),
                "MaxDD": r["metrics"].get("MaxDrawdown", 0.0),
                "WinRate": r["metrics"].get("WinRate", 0.0),
            })
        st.dataframe(pd.DataFrame(preview_rows))

        # Select runs to compare
        names = [r["name"] for r in runs]
        selected = st.multiselect("Select runs to compare", names, default=names[:min(3, len(names))])
        if selected:
            # Combine equity curves by index
            for r in runs:
                if r["name"] in selected:
                    eq = r["equity"].copy()
                    eq = eq.rename(columns={"equity": r["name"]})
                    combined = eq if combined is None else combined.join(eq, how="outer")

        # Plot comparison (interactive)
        if combined is not None and not combined.empty:
            fig_cmp = go.Figure()
            df_cmp = combined.interpolate().fillna(method="ffill").reset_index()
            for col in df_cmp.columns:
                if col == "date":
                    continue
                fig_cmp.add_trace(go.Scatter(x=df_cmp["date"], y=df_cmp[col], mode="lines", name=col))
            fig_cmp.update_layout(title="Equity Comparison", xaxis_title="date", yaxis_title="equity", margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Download combined CSV
            csv_bytes = combined.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Compared Equity CSV",
                data=csv_bytes,
                file_name="compare_runs_equity.csv",
                mime="text/csv",
            )

st.caption("© 2025 alireza-devtrader — devtrader-backtest")