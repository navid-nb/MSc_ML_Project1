"""Metrics helpers (no UI types)."""

from typing import Any, Dict


def extract_core_metrics(stats: Dict[str, Any]) -> Dict[str, float]:
    """Extract commonly used metrics with safe defaults.

    Parameters
    ----------
    stats : Dict[str, Any]
        Raw stats dict from backtesting.py.

    Returns
    -------
    Dict[str, float]
        Mapping of key metrics to numeric values.
    """

    def g(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            if k in stats and isinstance(stats[k], (int, float)):
                return float(stats[k])
        return float(default)

    return {
        "return_total_pct": g("Return [%]", "Return (Ann.) [%]"),  # prefer total if present
        "sharpe": g("Sharpe Ratio", "Sharpe"),
        "win_rate_pct": g("Win Rate [%]"),
        "max_drawdown_pct": g("Max. Drawdown [%]", "Max Drawdown [%]"),
        "trades": g("Trades"),
        "exposure_pct": g("Exposure [%]"),
        "equity_final": g("Equity Final [$]", "Equity Final [$ ]"),
    }
