"""Matplotlib plotter with equity-curve fallback."""

from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd

from backtester.interfaces.plotter import Plotter


class MatplotlibPlotter(Plotter):
    """If `backtesting_fig` is not an MPL fig, fallback to equity curve."""

    def render(self, backtesting_fig: Any, stats: Mapping[str, Any]) -> Any:
        """Return a matplotlib figure."""
        try:
            import matplotlib.figure as mpl_figure

            if isinstance(backtesting_fig, mpl_figure.Figure):
                return backtesting_fig
        except Exception:
            pass

        # Fallback: draw equity curve from stats
        eq = stats.get("_equity_curve")
        if isinstance(eq, pd.DataFrame) and "Equity" in eq.columns:
            fig, ax = plt.subplots()
            ax.plot(eq.index, eq["Equity"].values)
            ax.set_title("Equity Curve")
            ax.set_xlabel("Time")
            ax.set_ylabel("Equity")
            return fig

        # As a last resort, return what we have (frontends may handle it)
        return backtesting_fig
