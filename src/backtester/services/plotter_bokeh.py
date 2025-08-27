"""Bokeh plotter implementation."""

from typing import Any, Mapping

from backtester.interfaces.plotter import Plotter


class BokehPlotter(Plotter):
    """Return Bokeh figure directly (as produced by backtesting.py)."""

    def render(self, backtesting_fig: Any, stats: Mapping[str, Any]) -> Any:
        """Return the figure untouched for a Bokeh-capable frontend."""
        return backtesting_fig
