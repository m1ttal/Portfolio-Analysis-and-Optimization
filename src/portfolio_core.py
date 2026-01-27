"""
Core portfolio analytics & optimization functions.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Returns & Risk

def prepare_returns(prices, periods=252):
    """Log returns, annualized mean & covariance."""
    r = np.log(prices / prices.shift(1)).dropna()
    return r, r.mean() * periods, r.cov() * periods

# Portfolio Metrics

def portfolio_performance(w, mu, sigma, rf=0.0):
    """Return, volatility, Sharpe."""
    ret = w @ mu
    vol = np.sqrt(w @ sigma.values @ w)
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe

# Optimization Helpers

def _constraints(mu=None, target=None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mu - target})
    return cons


def _optimize(obj, n, cons):
    res = minimize(
        obj,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=cons,
    )
    return res.x, res.fun

# Random Portfolios

def simulate_random_portfolios(mu, sigma, n=5000, rf=0.0):
    """Monte Carlo portfolios."""
    w = np.random.dirichlet(np.ones(len(mu)), n)
    ret = w @ mu.values
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, sigma.values, w))
    return pd.DataFrame(
        {"return": ret, "volatility": vol, "sharpe": (ret - rf) / vol}
    )

# Efficient Frontier

def efficient_frontier(mu, sigma, points=100):
    n = len(mu)
    targets = np.linspace(mu.min(), mu.max(), points)

    data = [
        (t, _optimize(
            lambda w: np.sqrt(w @ sigma.values @ w),
            n,
            _constraints(mu, t)
        )[1])
        for t in targets
    ]

    df = pd.DataFrame(data, columns=["return", "volatility"])
    df = df.sort_values("volatility")
    df = df[df["return"] >= df["return"].cummax()]

    return df


# Optimal Portfolios

def global_min_variance(mu, sigma):
    """GMVP weights."""
    w, _ = _optimize(
        lambda w: np.sqrt(w @ sigma.values @ w),
        len(mu),
        _constraints(),
    )
    return w


def max_sharpe_ratio(mu, sigma, rf=0.0):
    """Tangency portfolio."""
    w, _ = _optimize(
        lambda w: -portfolio_performance(w, mu, sigma, rf)[2],
        len(mu),
        _constraints(),
    )
    return w
