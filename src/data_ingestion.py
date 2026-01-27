"""
DATA INGESTION & CLEANING PIPELINE

This script retrieves historical adjusted closing prices for selected stocks and a benchmark index using Yahoo Finance. 
It performs automated data validation by detecting and correcting outliers via Z-score filtering and handling missing values 
through forward-filling. 

- Fetches adjusted close prices via Yahoo Finance
- Forward-fills missing values
- Removes extreme outliers using Z-score
- Returns clean stock & benchmark data with metadata
"""

from pathlib import Path
import datetime
import platform

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

# Configuration

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

TICKERS = cfg["tickers"]
BENCHMARK = cfg["benchmark"]
YEARS = cfg["years"]

# Data Cleaning Utilities

def clean_outliers(df: pd.DataFrame, z_thresh: float = 3.0):
    """
    Replace extreme Z-score outliers with forward-filled values.
    """
    z = (df - df.mean()) / df.std()
    mask = z.abs() > z_thresh

    outlier_count = mask.sum().to_dict()

    return df.mask(mask).ffill(), outlier_count


def clean_prices(df: pd.DataFrame):
    """
    Apply standard cleaning steps:
    - Forward-fill missing values
    - Remove outliers
    """
    missing_before = int(df.isna().sum().sum())

    df = df.ffill()
    df, outliers = clean_outliers(df)

    return df, missing_before, outliers

# Main Ingestion Function

def fetch_data():
    """
    Fetch and clean historical adjusted close prices.
    """

    # Download adjusted prices
    prices = yf.download(
        TICKERS + [BENCHMARK],
        period=f"{YEARS}y",
        auto_adjust=True,
        progress=False
    )["Close"]

    # Split stock and benchmark data
    stocks_raw = prices[TICKERS]
    benchmark_raw = prices[[BENCHMARK]]

    # Clean datasets
    stocks, miss_s, out_s = clean_prices(stocks_raw)
    benchmark, miss_b, out_b = clean_prices(benchmark_raw)

    # Metadata
    metadata = {
        "run_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "python": platform.python_version(),
            "yfinance": yf.__version__,
        },
        "data_summary": {
            "rows": len(stocks),
            "date_range": (
                str(stocks.index.min().date()),
                str(stocks.index.max().date())
            ),
            "missing_values_before": miss_s + miss_b,
            "outliers_cleaned": sum(out_s.values()) + sum(out_b.values()),
        },
    }

    return stocks, benchmark, metadata
