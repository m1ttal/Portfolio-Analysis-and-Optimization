# Portfolio-Analysis-and-Optimization

This project analyzes financial assets to build and optimize portfolios using data-driven methods. Automated data ingestion retrieves and cleans stock and benchmark prices. Expected returns, variance, and correlations are calculated, and Markowitz portfolio optimization produces the efficient frontier and optimal portfolios.

# Objectives

Analyze historical stock price data

Compute returns, volatility, and correlation

Construct optimal portfolios using:

Global Minimum Variance Portfolio (GMVP)

Maximum Sharpe Ratio (Tangency Portfolio)

Visualize the Efficient Frontier

Evaluate asset risk using CAPM and Beta analysis

Compare portfolio performance against a benchmark

# Key Concepts Used

Log Returns & Annualization

Risk (Volatility) and Return Trade-off

Correlation & Diversification

Mean–Variance Optimization

Efficient Frontier

Sharpe Ratio

Capital Asset Pricing Model (CAPM)

Portfolio Weights Interpretation

# Tools & Technologies

Language: Python

Libraries:

NumPy

Pandas

Matplotlib

SciPy

Statsmodels

Environment: Jupyter Notebook

## Project Structure

```
Portfolio-Analysis-and-Optimization/
│
├── config/
│   └── config.yaml                   # Tickers, benchmark, time horizon
│
├── src/
│   ├── data_ingestion.py             # Data fetching & cleaning pipeline
│   └── portfolio_core.py             # Portfolio analytics & optimization
│
├── notebooks/
│   └── Portfolio_Analysis&Optimization.ipynb
│                                     # Main analysis notebook
│
├── reports/
│   └── project_report/               # Term Project Report in IEEE research paper format.
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── LICENSE                           # MIT License
└── .gitignore


```
