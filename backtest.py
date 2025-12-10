# backtest.py
"""
FinTech Factor Backtester - Strategy & Analytics

Reads daily prices from SQLite, runs a simple long-only
momentum + volatility filter strategy on a FinTech universe,
and compares performance vs a benchmark ETF.

Run:
    pip install pandas numpy sqlalchemy matplotlib
    python backtest.py
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# ----------------------- CONFIG -----------------------

DB_PATH = "fintech_backtest.db"
TABLE_NAME = "daily_prices"

UNIVERSE = [
    "SQ", "PYPL", "AFRM", "SOFI", "NU", "ADYEY",
    "UPST", "FISV", "FIS", "GPN", "HOOD", "AXP"
]

BENCHMARK = "XLF"

LOOKBACK_DAYS = 126     # ~6 months
VOL_WINDOW_DAYS = 60    # ~3 months
VOL_THRESHOLD = 0.04    # max daily volatility (filter)
TOP_N = 5               # number of names to hold each month
TRANS_COST_BPS = 10     # 10 bps per round-trip
RISK_FREE_RATE = 0.00   # assume 0 for simplicity

# ------------------------------------------------------


def load_price_data(db_path, table_name):
    engine = create_engine(f"sqlite:///{db_path}")
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine, parse_dates=["date"])

    # Pivot to wide format: index=date, columns=symbol, values=adj_close
    price = df.pivot(index="date", columns="symbol", values="adj_close")
    price = price.sort_index().ffill().dropna(how="all")
    return price


def compute_returns(price):
    """Compute simple daily returns from adjusted close prices."""
    rets = price.pct_change().dropna(how="all")
    return rets


def run_strategy(price, rets):
    """
    Long-only momentum + volatility filter strategy with monthly rebalancing.

    Rules:
      - At each month-end, compute 6-month momentum and 60-day volatility.
      - Keep only stocks under VOL_THRESHOLD.
      - Go equal-weight long top N by momentum for the next month.
      - Apply transaction costs at rebal dates based on turnover.
    """
    # Align to trading days, ensure universe+benchmark present
    available_cols = [c for c in UNIVERSE + [BENCHMARK] if c in price.columns]
    price = price[available_cols]
    rets = rets[available_cols]

    universe = [c for c in UNIVERSE if c in price.columns]
    bench = BENCHMARK

    # Month-end rebal dates (trading days)
    month_ends = rets.resample("M").last().index

    weights = pd.DataFrame(0.0, index=rets.index, columns=universe)
    prev_weights = pd.Series(0.0, index=universe)

    for date in month_ends:
        if date not in rets.index:
            # If month-end is not a trading day, skip
            continue

        # Window for momentum & volatility: up to 'date'
        end_loc = rets.index.get_loc(date)
        start_loc = max(0, end_loc - LOOKBACK_DAYS)
        window = rets.iloc[start_loc:end_loc + 1]

        if len(window) < LOOKBACK_DAYS // 2:
            # Not enough history yet
            continue

        # Compute momentum (~6m total return)
        # (1 + r).prod() - 1
        mom = (1.0 + window[universe]).prod() - 1.0

        # Compute volatility (daily) over VOL_WINDOW_DAYS
        vol_window = window.iloc[-VOL_WINDOW_DAYS:] if len(window) >= VOL_WINDOW_DAYS else window
        vol = vol_window[universe].std()

        # Filter by volatility
        eligible = mom[vol <= VOL_THRESHOLD].dropna()

        if eligible.empty:
            # No positions -> stay in cash
            curr_weights = pd.Series(0.0, index=universe)
        else:
            # Top N by momentum
            top = eligible.sort_values(ascending=False).head(TOP_N).index
            curr_weights = pd.Series(0.0, index=universe)
            curr_weights.loc[top] = 1.0 / len(top)

        # Store weights for this rebal date
        weights.loc[date] = curr_weights.values

        # Compute turnover for transaction costs
        turnover = (curr_weights - prev_weights).abs().sum()
        prev_weights = curr_weights

    # Forward-fill weights between rebalance dates
    weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)

    # Strategy daily returns: sum(weights * returns) across universe
    strat_rets_gross = (weights.shift(1) * rets[universe]).sum(axis=1)

    # Transaction costs at rebalance days
    # Recompute turnover day-by-day
    trans_cost_daily = pd.Series(0.0, index=strat_rets_gross.index)
    last_w = pd.Series(0.0, index=universe)

    for date in month_ends:
        if date not in weights.index:
            continue
        w = weights.loc[date]
        turnover = (w - last_w).abs().sum()
        cost = (TRANS_COST_BPS / 10000.0) * turnover
        trans_cost_daily.loc[date] = cost
        last_w = w

    strat_rets_net = strat_rets_gross - trans_cost_daily

    # Benchmark returns
    bench_rets = rets[bench]

    return strat_rets_net, bench_rets, weights


def compute_performance_metrics(rets, freq=252):
    """
    Compute standard risk/return metrics from a return series.
    rets: daily returns.
    """
    rets = rets.dropna()
    if rets.empty:
        return {
            "CAGR": np.nan,
            "AnnVol": np.nan,
            "Sharpe": np.nan,
            "MaxDrawdown": np.nan,
        }

    cum_return = (1.0 + rets).prod()
    num_days = len(rets)
    cagr = cum_return ** (freq / num_days) - 1.0

    ann_vol = rets.std() * np.sqrt(freq)
    sharpe = (rets.mean() * freq - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan

    # Compute max drawdown
    equity_curve = (1.0 + rets).cumprod()
    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1.0
    max_dd = drawdowns.min()

    return {
        "CAGR": cagr,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
    }


def plot_results(strat_rets, bench_rets):
    equity_strat = (1.0 + strat_rets).cumprod()
    equity_bench = (1.0 + bench_rets).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(equity_strat.index, equity_strat.values, label="Strategy")
    plt.plot(equity_bench.index, equity_bench.values, label=BENCHMARK)
    plt.title("Equity Curve: Strategy vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Drawdown plot
    running_max = equity_strat.cummax()
    drawdown = equity_strat / running_max - 1.0
    plt.figure(figsize=(10, 4))
    plt.plot(drawdown.index, drawdown.values)
    plt.title("Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    price = load_price_data(DB_PATH, TABLE_NAME)
    rets = compute_returns(price)

    strat_rets, bench_rets, weights = run_strategy(price, rets)

    strat_metrics = compute_performance_metrics(strat_rets)
    bench_metrics = compute_performance_metrics(bench_rets)

    print("=== Strategy Performance ===")
    for k, v in strat_metrics.items():
        print(f"{k}: {v:.3%}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== Benchmark Performance (XLF) ===")
    for k, v in bench_metrics.items():
        print(f"{k}: {v:.3%}" if isinstance(v, float) else f"{k}: {v}")

    plot_results(strat_rets, bench_rets)


if __name__ == "__main__":
    main()
