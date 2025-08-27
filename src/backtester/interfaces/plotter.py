"""Plotter interface."""

import abc
from typing import Any, Mapping


class Plotter(abc.ABC):
    """Abstract interface for rendering figures.

    Methods
    -------
    render(backtesting_fig, stats) -> Any
        Return a UI-framework-specific artifact (Bokeh figure, matplotlib fig, etc.).
    """

    @abc.abstractmethod
    def render(self, backtesting_fig: Any, stats: Mapping[str, Any]) -> Any:
        """Render a figure for the chosen frontend.

        Parameters
        ----------
        backtesting_fig : Any
            Figure returned by backtesting.py.
        stats : Mapping[str, Any]
            Stats dict from backtest run.

        Returns
        -------
        Any
            Renderable object for a given UI framework.
        """
        raise NotImplementedError
