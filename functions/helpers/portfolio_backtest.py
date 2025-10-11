"""
Portfolio Backtesting Utilities

This module provides functions for backtesting portfolio strategies
and calculating performance metrics.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def calculate_portfolio_returns(
    df: pd.DataFrame,
    weights_col: str = "portfolio_weights",
    returns_col: str = "adj_prc_logret_lead1",
    date_col: str = "date",
) -> pd.Series:
    """
    Calculate daily portfolio returns from individual stock weights and returns.

    Formula: port_ret[t] = Σ_i w_{i,t-1} * R_{i,t}

    Args:
        df: DataFrame with multi-index (stock_col, date_col) containing weights and returns
        weights_col: Column name for portfolio weights
        returns_col: Column name for stock returns
        date_col: Name of date index level

    Returns:
        Series of daily portfolio returns indexed by date
    """
    # Group by date and calculate weighted return for each day
    portfolio_returns = df.groupby(level=date_col).apply(
        lambda x: (x[weights_col] * x[returns_col]).sum()
    )

    return portfolio_returns


def calculate_equity_curve(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    """
    Calculate cumulative equity curve from daily returns.

    Formula: equity[t] = equity[t-1] * exp(ret[t])

    Args:
        returns: Series of portfolio returns
        initial_capital: Starting equity value

    Returns:
        Series of cumulative equity values
    """
    # Cumulative product of (1 + returns) for simple returns
    # Or cumulative sum of log returns then exp
    equity = initial_capital * np.exp(returns.cumsum())
    return equity


def calculate_performance_metrics(
    returns: pd.Series, rf_rate: float = 0.0, periods_per_year: int = 252
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics for a return series.

    Args:
        returns: Series of portfolio returns
        rf_rate: Risk-free rate (annualized)
        periods_per_year: Number of trading periods per year (252 for daily)

    Returns:
        Dictionary of performance metrics
    """
    # Basic statistics
    n_periods = len(returns)
    mean_return = returns.mean()
    std_return = returns.std()

    # Annualized metrics
    ann_return = mean_return * periods_per_year
    ann_vol = std_return * np.sqrt(periods_per_year)

    # Sharpe ratio
    sharpe = (ann_return - rf_rate) / ann_vol if ann_vol > 0 else 0.0

    # Cumulative return
    total_return = np.exp(returns.sum()) - 1

    # Max drawdown
    equity = np.exp(returns.cumsum())
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (returns > 0).mean()

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino = (
        (ann_return - rf_rate) / (downside_std * np.sqrt(periods_per_year))
        if downside_std > 0
        else 0.0
    )

    # Calmar ratio (return / max drawdown)
    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    return {
        "n_periods": n_periods,
        "total_return": total_return * 100,  # as percentage
        "ann_return": ann_return * 100,  # as percentage
        "ann_vol": ann_vol * 100,  # as percentage
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown * 100,  # as percentage
        "win_rate": win_rate * 100,  # as percentage
        "avg_daily_ret": mean_return * 100,  # as percentage
        "daily_vol": std_return * 100,  # as percentage
    }


def backtest_strategy(
    df: pd.DataFrame,
    scoring_method: str,
    allocation_strategy: str,
    score_columns: Dict[str, str],
    allocation_func,
    allocation_params: Dict[str, Any],
    returns_col: str = "adj_prc_logret_lead1",
    date_col: str = "date",
    stock_col: str = "permno",
    rf_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Backtest a complete strategy (scoring + allocation) on given data.

    Args:
        df: DataFrame with all necessary data (scores, returns, etc.)
        scoring_method: Name of scoring method (e.g., "S1", "S2")
        allocation_strategy: Name of allocation strategy (e.g., "A1", "A2")
        score_columns: Dict mapping scoring method names to column names
        allocation_func: Function to apply allocation strategy
        allocation_params: Parameters for allocation function
        returns_col: Column name for stock returns
        date_col: Name of date index level
        stock_col: Name of stock index level
        rf_rate: Risk-free rate for Sharpe calculation

    Returns:
        Dictionary with strategy results and metrics
    """
    # Get the score column for this method
    score_col = score_columns.get(scoring_method)
    if score_col not in df.columns:
        raise ValueError(f"Score column {score_col} not found in DataFrame")

    # Create a working copy
    df_work = df.copy()

    # Apply allocation strategy to get weights
    # Need to apply per-date
    weights_list = []
    dates = df_work.index.get_level_values(date_col).unique()

    for date in dates:
        date_df = df_work.loc[df_work.index.get_level_values(date_col) == date]

        # Get scores and masks for this date
        scores = date_df[score_col]
        long_mask = date_df["agreed_long"]
        short_mask = date_df["agreed_short"]
        volatility = date_df.get("volatility", pd.Series(1.0, index=date_df.index))
        expected_returns = date_df.get("expected_return", pd.Series(0.0, index=date_df.index))

        # Apply allocation
        try:
            weights = allocation_func(
                strategy_name=allocation_strategy,
                scores=scores,
                long_mask=long_mask,
                short_mask=short_mask,
                volatility=volatility,
                expected_returns=expected_returns,
                **allocation_params,
            )
            weights_list.append(weights)
        except Exception:
            # If allocation fails, use zero weights for this date
            weights_list.append(pd.Series(0.0, index=date_df.index))

    # Combine all weights
    df_work["portfolio_weights"] = pd.concat(weights_list)

    # Calculate returns
    portfolio_returns = calculate_portfolio_returns(
        df_work,
        weights_col="portfolio_weights",
        returns_col=returns_col,
        date_col=date_col,
    )

    # Calculate metrics
    metrics = calculate_performance_metrics(portfolio_returns, rf_rate=rf_rate)

    # Calculate equity curve
    equity = calculate_equity_curve(portfolio_returns)

    return {
        "scoring_method": scoring_method,
        "allocation_strategy": allocation_strategy,
        "metrics": metrics,
        "returns": portfolio_returns,
        "equity": equity,
        "weights": df_work["portfolio_weights"],
    }
