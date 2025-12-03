"""
DATA INGESTION PIPELINE

This script retrieves historical adjusted closing prices for selected stocks and a benchmark index using Yahoo Finance. 
It performs automated data validation by detecting and correcting outliers via Z-score filtering and handling missing values 
through forward-filling. 

The cleaned datasets are exported as processed CSV files, and a metadata JSON file is generated to capture configuration inputs, 
software environment details, cleaning statistics, and data quality metrics. This ensures full reproducibility and prepares the 
data for subsequent financial modeling, risk analysis, and portfolio optimization workflows.
"""

import os
import json
import datetime
import platform
import pandas as pd
import numpy as np
import yfinance as yf
import yaml

# Load YAML config
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

TICKERS   = cfg["tickers"]
BENCHMARK = cfg["benchmark"]
YEARS     = cfg["years"]
PATHS     = cfg["paths"]

# function to clean outliers
def clean_outliers(df):
    """
    Detects outliers using Z-score method and replaces them with forward-filled values.
    
    Args:
        df (pd.DataFrame): DataFrame containing price data.
    
    Returns:
        cleaned_df (pd.DataFrame): DataFrame with outliers replaced.
        outlier_details (dict): Number of outliers per column.
    """
    z_scores = (df - df.mean()) / df.std()
    mask = np.abs(z_scores) > 3
    outlier_details = mask.sum(axis=0).to_dict()
    
    cleaned_df = df.where(~mask, np.nan).ffill()
    return cleaned_df, {k: int(v) for k, v in outlier_details.items()}

# Main data ingestion pipeline
def fetch_data():
    """
    Fetches adjusted close prices for tickers and benchmark,
    cleans outliers, forward-fills missing values,
    saves processed CSVs and metadata JSON.
    
    Returns:
        stocks (pd.DataFrame): Cleaned stock data.
        benchmark (pd.DataFrame): Cleaned benchmark data.
        metadata (dict): Dictionary containing metadata of the run.
    """
    # Download adjusted close prices
    data = yf.download(TICKERS + [BENCHMARK], period=f"{YEARS}y", auto_adjust=True)['Close']

    # Separate stock and benchmark data and forward-fill missing values
    stocks = data[TICKERS].ffill()
    benchmark = data[[BENCHMARK]].ffill()
    missing_before = int(stocks.isna().sum().sum() + benchmark.isna().sum().sum())

    # Clean outliers
    stocks, details_stock = clean_outliers(stocks)
    benchmark, details_bench = clean_outliers(benchmark)
    total_outliers = sum(details_stock.values()) + sum(details_bench.values())

    
    # Prepare file names with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    stock_file = os.path.join(PATHS["processed"], f"stocks_adj_{timestamp}.csv")
    benchmark_file = os.path.join(PATHS["processed"], f"benchmark_adj_{timestamp}.csv")
    
    # Save processed CSV files
    stocks.to_csv(stock_file)
    benchmark.to_csv(benchmark_file)

    # Prepare metadata
    metadata = {
        "run_info": {
            "timestamp": timestamp,
            "python_version": platform.python_version(),
            "yfinance_version": yf.__version__
        },
        "config": cfg,
        "data_summary": {
            "rows": len(stocks),
            "start_date": str(stocks.index.min().date()),
            "end_date": str(stocks.index.max().date()),
            "missing_values_before_cleaning": missing_before,
            "outliers_cleaned": int(total_outliers),
            "outliers_detail": {
                "stocks": details_stock,
                "benchmark": details_bench
            }
        },
        "file_paths": {
            "stocks_file": stock_file,
            "benchmark_file": benchmark_file
        }
    }
    
    # Save metadata JSON
    metadata_file = os.path.join(PATHS["metadata"], f"metadata_{timestamp}.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    
    return stocks, benchmark, metadata

# Execute pipeline
if __name__ == "__main__":
    stocks, benchmark, metadata = fetch_data()
    print("Sample Stock Data:")
    print(stocks.head())
    print("\nSample Benchmark Data:")
    print(benchmark.head())

