# data_pipeline.py
"""
FinTech Factor Backtester - Data ETL

1. Define a FinTech/FinTech-adjacent universe and a benchmark.
2. Download daily OHLCV data using yfinance.
3. Store cleaned price data into a local SQLite database.

Run:
    pip install yfinance pandas sqlalchemy
    python data_pipeline.py
"""

import datetime as dt
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine

# ----------------------- CONFIG -----------------------

UNIVERSE = [
    "SQ",   # Block
    "PYPL", # PayPal
    "AFRM", # Affirm
    "SOFI", # SoFi
    "NU",   # Nubank
    "ADYEY",# Adyen (ADR)
    "UPST", # Upstart
    "FISV", # Fiserv
    "FIS",  # Fidelity National Information Services
    "GPN",  # Global Payments
    "HOOD", # Robinhood
    "AXP",  # American Express (FinTech-adjacent)
]

BENCHMARK = "XLF"  # Financials ETF (could also use VFH, SPY, etc.)

START_DATE = "2015-01-01"
END_DATE = dt.date.today().strftime("%Y-%m-%d")

DB_PATH = "fintech_backtest.db"
TABLE_NAME = "daily_prices"

# ------------------------------------------------------


def download_price_data(tickers, start, end):
    """
    Download daily OHLCV data for a list of tickers using yfinance.
    Returns a tidy DataFrame with columns:
        [date, symbol, open, high, low, close, adj_close, volume]
    """
    print(f"Downloading data for: {tickers}")
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        interval="1d",
    )

    if isinstance(raw.columns, pd.MultiIndex):
        frames = []
        for symbol in tickers:
            df_sym = raw[symbol].copy()
            df_sym["symbol"] = symbol
            frames.append(df_sym.reset_index())
        df = pd.concat(frames, ignore_index=True)
    else:
        # Single ticker case
        df = raw.copy()
        df["symbol"] = tickers[0]
        df = df.reset_index()

    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    # Ensure types
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Drop rows with no trading data
    df = df.dropna(subset=["close"])

    return df


def write_to_sqlite(df, db_path, table_name):
    """
    Write the tidy price DataFrame into a SQLite DB.
    If the table exists, we replace it.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Saved {len(df):,} rows to {db_path}:{table_name}")


def main():
    tickers = UNIVERSE + [BENCHMARK]
    df = download_price_data(tickers, START_DATE, END_DATE)
    write_to_sqlite(df, DB_PATH, TABLE_NAME)
    print("Data pipeline completed.")


if __name__ == "__main__":
    main()
