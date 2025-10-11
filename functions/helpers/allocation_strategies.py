"""
Portfolio Allocation Strategies

This module implements 9 different portfolio allocation strategies that convert
stock signals/scores into portfolio weights subject to constraints.

All strategies return a pandas Series of weights with the same index as the input.
"""

from typing import Optional

import numpy as np
import pandas as pd


def normalize_weights_dollar_neutral(
    weights: pd.Series, long_target: float = 0.5, short_target: float = 0.5
) -> pd.Series:
    """
    Normalize weights to achieve dollar neutrality.

    Args:
        weights: Series of portfolio weights
        long_target: Target long exposure (default 0.5 = 50%)
        short_target: Target short exposure (default 0.5 = 50%)

    Returns:
        Normalized weights
    """
    pos_mask = weights > 0
    neg_mask = weights < 0

    pos_sum = weights[pos_mask].sum()
    neg_sum = -weights[neg_mask].sum()

    weights_normalized = weights.copy()
    if pos_sum > 0:
        weights_normalized[pos_mask] = weights[pos_mask] / pos_sum * long_target
    if neg_sum > 0:
        weights_normalized[neg_mask] = weights[neg_mask] / neg_sum * (-short_target)

    return weights_normalized


def cap_position_sizes(weights: pd.Series, max_weight: float) -> pd.Series:
    """
    Cap individual position sizes and renormalize.

    Args:
        weights: Series of portfolio weights
        max_weight: Maximum absolute weight per position

    Returns:
        Capped and renormalized weights
    """
    weights_capped = weights.clip(lower=-max_weight, upper=max_weight)
    return normalize_weights_dollar_neutral(weights_capped)


def strategy_a1_equal_weighted(
    scores: pd.Series,
    long_mask: pd.Series,
    short_mask: pd.Series,
    long_target: float = 0.5,
    short_target: float = 0.5,
    max_position_size: float = 0.10,
) -> pd.Series:
    """
    A1: Equal-Weighted (EW) Portfolio

    Simplest benchmark; every selected stock gets the same weight.

    Args:
        scores: Stock scores (unused, but kept for consistent signature)
        long_mask: Boolean mask for long positions
        short_mask: Boolean mask for short positions
        long_target: Target long exposure
        short_target: Target short exposure
        max_position_size: Maximum weight per position

    Returns:
        Series of portfolio weights
    """
    weights = pd.Series(0.0, index=scores.index)

    n_long = long_mask.sum()
    n_short = short_mask.sum()

    if n_long > 0:
        weights[long_mask] = long_target / n_long
    if n_short > 0:
        weights[short_mask] = -short_target / n_short

    return cap_position_sizes(weights, max_position_size)


def strategy_a2_rank_weighted(
    scores: pd.Series,
    long_mask: pd.Series,
    short_mask: pd.Series,
    long_target: float = 0.5,
    short_target: float = 0.5,
    max_position_size: float = 0.10,
) -> pd.Series:
    """
    A2: Rank-Weighted (Signal-Weighted) Portfolio

    Weights proportional to model confidence or combined score.

    Args:
        scores: Stock scores
        long_mask: Boolean mask for long positions
        short_mask: Boolean mask for short positions
        long_target: Target long exposure
        short_target: Target short exposure
        max_position_size: Maximum weight per position

    Returns:
        Series of portfolio weights
    """
    weights = pd.Series(0.0, index=scores.index)
    scores_abs = scores.abs()

    # Long side
    long_scores = scores_abs[long_mask]
    if long_scores.sum() > 0:
        weights[long_mask] = (long_scores / long_scores.sum()) * long_target

    # Short side (negative weights)
    short_scores = scores_abs[short_mask]
    if short_scores.sum() > 0:
        weights[short_mask] = -(short_scores / short_scores.sum()) * short_target

    return cap_position_sizes(weights, max_position_size)


def strategy_a3_quantile(
    scores: pd.Series,
    long_mask: pd.Series,
    short_mask: pd.Series,
    long_target: float = 0.5,
    short_target: float = 0.5,
    max_position_size: float = 0.10,
    quantile_long_pct: float = 0.20,
    quantile_short_pct: float = 0.20,
) -> pd.Series:
    """
    A3: Top/Bottom Quantile Strategy

    Long/short based on signal ranks.
    - Go long top x% (e.g., top 10–20%)
    - Go short bottom x% (e.g., bottom 10–20%)

    Args:
        scores: Stock scores
        long_mask: Boolean mask for long positions (unused, quantile determines selection)
        short_mask: Boolean mask for short positions (unused, quantile determines selection)
        long_target: Target long exposure
        short_target: Target short exposure
        max_position_size: Maximum weight per position
        quantile_long_pct: Percentage for top quantile
        quantile_short_pct: Percentage for bottom quantile

    Returns:
        Series of portfolio weights
    """
    weights = pd.Series(0.0, index=scores.index)

    # Determine quantile thresholds
    quantile_long = scores.quantile(1 - quantile_long_pct)
    quantile_short = scores.quantile(quantile_short_pct)

    # Select top and bottom
    agreed_mask = long_mask | short_mask
    top_mask = (scores >= quantile_long) & agreed_mask
    bottom_mask = (scores <= quantile_short) & agreed_mask

    n_top = top_mask.sum()
    n_bottom = bottom_mask.sum()

    if n_top > 0:
        weights[top_mask] = long_target / n_top
    if n_bottom > 0:
        weights[bottom_mask] = -short_target / n_bottom

    return cap_position_sizes(weights, max_position_size)


def strategy_a4_long_only(
    scores: pd.Series,
    long_mask: pd.Series,
    short_mask: pd.Series,
    long_target: float = 1.0,
    short_target: float = 0.0,
    max_position_size: float = 0.10,
    threshold_percentile: float = 0.60,
) -> pd.Series:
    """
    A4: Long-Only Threshold Portfolio

    Only go long when predicted return or probability is above a cutoff.

    Args:
        scores: Stock scores
        long_mask: Boolean mask for long positions
        short_mask: Boolean mask for short positions (unused)
        long_target: Target long exposure (typically 1.0 for long-only)
        short_target: Target short exposure (typically 0.0)
        max_position_size: Maximum weight per position
        threshold_percentile: Only trade above this percentile

    Returns:
        Series of portfolio weights
    """
    weights = pd.Series(0.0, index=scores.index)

    # Determine threshold
    threshold = scores.quantile(threshold_percentile)
    agreed_mask = long_mask | short_mask
    above_threshold = (scores >= threshold) & agreed_mask

    n_above = above_threshold.sum()
    if n_above > 0:
        weights[above_threshold] = long_target / n_above

    return cap_position_sizes(weights, max_position_size)


def strategy_a5_volatility_scaled(
    scores: pd.Series,
    long_mask: pd.Series,
    short_mask: pd.Series,
    volatility: pd.Series,
    long_target: float = 0.5,
    short_target: float = 0.5,
    max_position_size: float = 0.10,
) -> pd.Series:
    """
    A5: Volatility-Scaled Allocation (Inverse-Vol Weighting)

    More weight to stable assets, less to volatile ones.

    Args:
        scores: Stock scores
        long_mask: Boolean mask for long positions
        short_mask: Boolean mask for short positions
        volatility: Stock volatilities
        long_target: Target long exposure
        short_target: Target short exposure
        max_position_size: Maximum weight per position

    Returns:
        Series of portfolio weights
    """
    weights = pd.Series(0.0, index=scores.index)

    # Weight proportional to score / volatility
    risk_adjusted = scores / (volatility + 1e-8)
    risk_adjusted_abs = risk_adjusted.abs()

    # Long side
    long_scores = risk_adjusted_abs[long_mask]
    if long_scores.sum() > 0:
        weights[long_mask] = (long_scores / long_scores.sum()) * long_target

    # Short side
    short_scores = risk_adjusted_abs[short_mask]
    if short_scores.sum() > 0:
        weights[short_mask] = -(short_scores / short_scores.sum()) * short_target

    return cap_position_sizes(weights, max_position_size)


def strategy_a6_max_sharpe(
    scores: pd.Series,
    long_mask: pd.Series,
    short_mask: pd.Series,
    returns_df: pd.DataFrame,
    long_target: float = 0.5,
    short_target: float = 0.5,
    max_position_size: float = 0.10,
) -> pd.Series:
    """
    A6: Maximum Sharpe Ratio Portfolio (Mean–Variance Optimization)

    Optimize weights to maximize expected Sharpe ratio given model forecasts.

    Args:
        scores: Stock scores (used as expected returns)
        long_mask: Boolean mask for long positions
        short_mask: Boolean mask for short positions
        returns_df: DataFrame of historical returns for covariance estimation
        long_target: Target long exposure
        short_target: Target short exposure
        max_position_size: Maximum weight per position

    Returns:
        Series of portfolio weights
    """
    weights = pd.Series(0.0, index=scores.index)
    agreed_mask = long_mask | short_mask

    try:
        from scipy.optimize import minimize

        # Get agreed stocks
        agreed_indices = agreed_mask[agreed_mask].index
        n_stocks = len(agreed_indices)

        if n_stocks > 1:
            # Expected returns (use scores as proxy)
            mu = scores[agreed_indices].values

            # Covariance matrix from returns
            cov_matrix = (
                returns_df.loc[agreed_indices].unstack(level="permno").fillna(0).cov().values
            )

            # Add regularization to prevent singular matrix
            cov_matrix += np.eye(n_stocks) * 1e-4

            # Objective: negative Sharpe
            def neg_sharpe(w):
                port_return = w @ mu
                port_vol = np.sqrt(w @ cov_matrix @ w)
                return -port_return / (port_vol + 1e-8)

            # Constraints: sum to 0 (dollar neutral)
            constraints = [{"type": "eq", "fun": lambda w: w.sum()}]
            bounds = [(-max_position_size, max_position_size)] * n_stocks

            # Initial guess
            w0 = np.zeros(n_stocks)
            w0[: n_stocks // 2] = long_target / (n_stocks // 2)
            w0[n_stocks // 2 :] = -short_target / (n_stocks - n_stocks // 2)

            # Optimize
            result = minimize(
                neg_sharpe,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 100},
            )

            if result.success:
                weights[agreed_indices] = result.x
            else:
                # Fallback to equal weight
                return strategy_a1_equal_weighted(
                    scores, long_mask, short_mask, long_target, short_target, max_position_size
                )
        else:
            # Fallback to equal weight
            return strategy_a1_equal_weighted(
                scores, long_mask, short_mask, long_target, short_target, max_position_size
            )
    except (Exception,):
        # Fallback if optimization fails
        return strategy_a1_equal_weighted(
            scores, long_mask, short_mask, long_target, short_target, max_position_size
        )

    return cap_position_sizes(weights, max_position_size)


def strategy_a7_risk_parity(
    scores: pd.Series,
    long_mask: pd.Series,
    short_mask: pd.Series,
    volatility: pd.Series,
    long_target: float = 0.5,
    short_target: float = 0.5,
    max_position_size: float = 0.10,
) -> pd.Series:
    """
    A7: Risk Parity / Equal Risk Contribution (ERC)

    Allocate so each position contributes equally to portfolio volatility.
    Simplified version: weight proportional to 1/volatility.

    Args:
        scores: Stock scores (unused, but kept for consistent signature)
        long_mask: Boolean mask for long positions
        short_mask: Boolean mask for short positions
        volatility: Stock volatilities
        long_target: Target long exposure
        short_target: Target short exposure
        max_position_size: Maximum weight per position

    Returns:
        Series of portfolio weights
    """
    weights = pd.Series(0.0, index=scores.index)

    # Weight proportional to 1/volatility
    inv_vol = 1 / (volatility + 1e-8)

    # Long side
    long_inv_vol = inv_vol[long_mask]
    if long_inv_vol.sum() > 0:
        weights[long_mask] = (long_inv_vol / long_inv_vol.sum()) * long_target

    # Short side
    short_inv_vol = inv_vol[short_mask]
    if short_inv_vol.sum() > 0:
        weights[short_mask] = -(short_inv_vol / short_inv_vol.sum()) * short_target

    return cap_position_sizes(weights, max_position_size)


def strategy_a8_softmax(
    scores: pd.Series,
    long_mask: pd.Series,
    short_mask: pd.Series,
    long_target: float = 0.5,
    short_target: float = 0.5,
    max_position_size: float = 0.10,
    lambda_param: float = 2.0,
) -> pd.Series:
    """
    A8: Softmax (Exponential Score) Allocation

    Convert scores to probability-like weights via exponential mapping.

    Args:
        scores: Stock scores
        long_mask: Boolean mask for long positions
        short_mask: Boolean mask for short positions
        long_target: Target long exposure
        short_target: Target short exposure
        max_position_size: Maximum weight per position
        lambda_param: Temperature parameter (higher = more aggressive)

    Returns:
        Series of portfolio weights
    """
    weights = pd.Series(0.0, index=scores.index)

    # Softmax on scores
    exp_scores = np.exp(lambda_param * scores)

    # Long side
    long_exp = exp_scores[long_mask]
    if long_exp.sum() > 0:
        weights[long_mask] = (long_exp / long_exp.sum()) * long_target

    # Short side
    short_exp = exp_scores[short_mask]
    if short_exp.sum() > 0:
        weights[short_mask] = -(short_exp / short_exp.sum()) * short_target

    return cap_position_sizes(weights, max_position_size)


def strategy_a9_kelly(
    scores: pd.Series,
    long_mask: pd.Series,
    short_mask: pd.Series,
    expected_returns: pd.Series,
    volatility: pd.Series,
    long_target: float = 0.5,
    short_target: float = 0.5,
    max_position_size: float = 0.10,
    kelly_fraction: float = 0.25,
) -> pd.Series:
    """
    A9: Kelly Criterion / Fractional Kelly Allocation

    Optimal capital fraction per asset under log-utility.
    w = f * (μ / σ²) where f is the Kelly fraction.

    Args:
        scores: Stock scores (unused, kelly uses expected_returns)
        long_mask: Boolean mask for long positions
        short_mask: Boolean mask for short positions
        expected_returns: Expected returns (from model)
        volatility: Stock volatilities
        long_target: Target long exposure
        short_target: Target short exposure
        max_position_size: Maximum weight per position
        kelly_fraction: Fraction of full Kelly (0.1-0.5 typical)

    Returns:
        Series of portfolio weights
    """
    weights = pd.Series(0.0, index=scores.index)

    # Kelly formula: w = f * (mu / sigma²)
    sigma_sq = volatility**2
    kelly_weights = kelly_fraction * (expected_returns / (sigma_sq + 1e-8))
    kelly_weights = kelly_weights.clip(lower=-max_position_size, upper=max_position_size)

    # Apply to agreed stocks only
    agreed_mask = long_mask | short_mask
    weights[agreed_mask] = kelly_weights[agreed_mask]

    # Normalize to dollar neutral
    weights = normalize_weights_dollar_neutral(weights, long_target, short_target)

    return cap_position_sizes(weights, max_position_size)


def apply_allocation_strategy(
    strategy_name: str,
    scores: pd.Series,
    long_mask: pd.Series,
    short_mask: pd.Series,
    volatility: Optional[pd.Series] = None,
    expected_returns: Optional[pd.Series] = None,
    returns_df: Optional[pd.DataFrame] = None,
    long_target: float = 0.5,
    short_target: float = 0.5,
    max_position_size: float = 0.10,
    **strategy_params,
) -> pd.Series:
    """
    Dispatcher function to apply the specified allocation strategy.

    Args:
        strategy_name: Strategy identifier ("A1" through "A9")
        scores: Stock scores
        long_mask: Boolean mask for long positions
        short_mask: Boolean mask for short positions
        volatility: Stock volatilities (required for A5, A7, A9)
        expected_returns: Expected returns (required for A9)
        returns_df: Historical returns DataFrame (required for A6)
        long_target: Target long exposure
        short_target: Target short exposure
        max_position_size: Maximum weight per position
        **strategy_params: Additional strategy-specific parameters

    Returns:
        Series of portfolio weights
    """
    strategy_map = {
        "A1": strategy_a1_equal_weighted,
        "A2": strategy_a2_rank_weighted,
        "A3": strategy_a3_quantile,
        "A4": strategy_a4_long_only,
        "A5": strategy_a5_volatility_scaled,
        "A6": strategy_a6_max_sharpe,
        "A7": strategy_a7_risk_parity,
        "A8": strategy_a8_softmax,
        "A9": strategy_a9_kelly,
    }

    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}. Must be A1-A9.")

    # Prepare common arguments
    common_args = {
        "scores": scores,
        "long_mask": long_mask,
        "short_mask": short_mask,
        "long_target": long_target,
        "short_target": short_target,
        "max_position_size": max_position_size,
    }

    # Add strategy-specific arguments
    if strategy_name in ["A5", "A7", "A9"] and volatility is not None:
        common_args["volatility"] = volatility
    if strategy_name == "A9" and expected_returns is not None:
        common_args["expected_returns"] = expected_returns
    if strategy_name == "A6" and returns_df is not None:
        common_args["returns_df"] = returns_df

    # Merge with additional parameters
    common_args.update(strategy_params)

    # Call the strategy function
    return strategy_map[strategy_name](**common_args)
