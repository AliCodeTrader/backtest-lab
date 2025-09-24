# backtest-lab 🧪📈  
> An interactive backtesting lab with Streamlit — test, optimize, and compare trading strategies fast.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-ff69b4.svg)](#)

A clean, plug-and-play framework for retail quants and dev-traders:
- Pluggable **strategies** (EMA+RSI, MA Crossover, MACD)
- **Short selling** (−1 / 0 / +1)
- **Position sizing** (fixed % or ATR risk-based)
- **SL/TP**, fees & slippage
- **Grid Search**, **Walk-Forward**, **Monte Carlo**
- **Compare Runs** & one-click **HTML report**
- Interactive **Plotly** charts, Streamlit dashboard

---

## 🚀 Quick Start

```bash
# clone
git clone https://github.com/AliCodeTrader/backtest-lab.git
cd backtest-lab

# create env
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install deps
pip install -r requirements.txt  # یا: pip install -U streamlit pandas numpy yfinance plotly matplotlib

# run app
streamlit run app.py
```

App UI at: `http://localhost:8501`

---

## 🧩 Features

- **Strategies (pluggable):**
  - EMA + RSI band filter
  - MA Crossover (EMA/SMA)
  - MACD (line/signal crossover)
- **Execution & Risk:**
  - Long/Short, position sizing (fixed / ATR-risk), SL/TP
  - Costs on **exposure changes** (fee + slippage)
- **Analytics:**
  - Equity, Drawdown, CAGR, Sharpe, Win Rate, Total Return
  - Grid Search with top-N table + heatmap
  - Walk-Forward (rolling/expanding) with OOS equity
  - Monte Carlo (iid / block bootstrap) + VaR/CVaR
- **Workflow:**
  - Save & Compare Runs (multi-line equity compare)
  - Export **HTML report** (params + metrics + chart)
  - CSV import or live **yfinance** loader

---

## 🖥️ Streamlit App (Tabs)

- **Parameters:** data source, strategy params, shorts, fees/slippage, SL/TP, sizing.
- **Metrics:** equity & stats (+ Walk-Forward, Monte Carlo).
- **Grid Search:** sweep EMA/RSI (fast < slow), top-N & heatmap.
- **Compare Runs:** save any run, select and plot interactively.
- **Reports:** export self-contained HTML report.

> Theme tip: create `.streamlit/config.toml`:
```toml
[theme]
base="dark"
primaryColor="#00c3ff"
backgroundColor="#0f1117"
secondaryBackgroundColor="#151a22"
textColor="#ffffff"
font="sans serif"
```

---

## 🧪 CLI Usage

Backtest from terminal (no UI):

```bash
# EMA+RSI (default)
python cli.py --ticker BTC-USD --start 2022-01-01   --strategy ema_rsi --fast 20 --slow 50 --rsi 14   --size-mode risk --risk-frac 0.02 --atr-len 14 --atr-mult 2.0   --allow-shorts --fee 0.0005 --slippage 0.0002 --sl 0.05 --tp 0.12
```

Other strategies:
```bash
# MA Crossover
python cli.py --ticker BTC-USD --strategy ma_cross --fast 10 --slow 40

# MACD (use --rsi as signal length)
python cli.py --ticker BTC-USD --strategy macd --fast 12 --slow 26 --rsi 9
```

Outputs include metrics + `outputs/equity_<TICKER>.png`.

---

## 🧠 How it works (TL;DR)

- Strategies produce `signal ∈ {-1,0,+1}`.
- Backtester keeps **entry size** fixed until exit/flip.
- **Risk sizing (ATR):** `size ≈ risk_frac / (atr_mult * ATR / price)` clipped to `[0,1]`.
- Costs apply on `Δ exposure = Δ(direction * size)`.
- SL/TP use entry-anchored cumulative P&L (long: `price/entry−1`, short: `entry/price−1`).

---

## 📂 Project Structure

```
backtest-lab/
├─ app.py                      # Streamlit dashboard (tabs + Plotly)
├─ cli.py                      # Command-line backtests
├─ requirements.txt
├─ src/devtrader/
│  ├─ data.py                  # CSV / yfinance loaders
│  ├─ indicators.py            # EMA, RSI, ATR
│  ├─ strategies.py            # EMA+RSI, MA Cross, MACD (+ registry)
│  ├─ strategy.py              # legacy EMA+RSI (used in grid search helper)
│  ├─ backtester.py            # sizing, shorts, SL/TP, costs
│  ├─ metrics.py               # equity metrics
│  ├─ plotting.py              # Plotly equity+DD figure
│  ├─ walkforward.py           # train/test folds & OOS
│  └─ montecarlo.py            # iid/block bootstrap + metrics
└─ outputs/                    # saved images/exports
```

---

## 📈 Screenshots (placeholders)

- `assets/screenshot_parameters.png`  
- `assets/screenshot_metrics.png`  
- `assets/screenshot_grid.png`  
- `assets/screenshot_compare.png`  
- `assets/screenshot_report.png`

*(Add images to `assets/` and link here.)*

---

## 🔬 Roadmap

- [x] Short selling (±1)
- [x] Position sizing (fixed / ATR risk)
- [x] Grid Search (EMA+RSI)
- [x] Walk-Forward (OOS)
- [x] Monte Carlo (iid / block)
- [x] Reports (HTML export)
- [ ] PDF export (pdfkit + wkhtmltopdf)
- [ ] Grid/WF for MA Crossover & MACD
- [ ] Strategy notebook mode (inline code in UI)
- [ ] More indicators (Bollinger, Keltner, ADX …)

---

## ⚖️ License

MIT © 2025 **AliCodeTrader**  
See [`LICENSE`](LICENSE).

---

