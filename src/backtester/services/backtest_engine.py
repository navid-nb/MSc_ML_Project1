"""Backtest engine (framework-agnostic)."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from loguru import logger

from backtester.interfaces.feature_engineer import FeatureEngineer
from backtester.interfaces.model import Model


@dataclass
class TrainTestConfig:
    """Train/test split and thresholds."""

    split_ratio: float = 0.7
    threshold: float = 0.5


@dataclass
class ExecConfig:
    """Execution config for backtesting."""

    cash: float = 10_000.0
    commission: float = 0.0005
    allow_short: bool = True
    finalize_trades: bool = True


class BacktestEngine:
    """Feature → model → signal → backtesting.py orchestration."""

    def __init__(self, feature_engineer: FeatureEngineer) -> None:
        """Construct the engine.

        Parameters
        ----------
        feature_engineer : FeatureEngineerInterface
            Feature builder instance.
        """
        self.fe = feature_engineer

    @staticmethod
    def _proba_to_signal(proba: Iterable[float], threshold: float) -> np.ndarray:
        """Map probability to trading signal {-1.0, +1.0}."""
        p = np.asarray(list(proba), dtype=float)
        return np.where(p >= threshold, 1.0, -1.0)

    def _build_signal_series(
        self, feats: pd.DataFrame, model: Model, cfg: TrainTestConfig
    ) -> Tuple[pd.Series, pd.DatetimeIndex]:
        """Train on first split and create signal on test part."""
        split_idx = int(len(feats) * cfg.split_ratio)
        X = feats.drop(columns=["y"])
        y = (feats["y"] > 0).astype(int)

        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        idx_test = X_test.index

        logger.info(f"Training {model.name} on {len(X_train)} rows; testing on {len(X_test)} rows")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        sig = pd.Series(self._proba_to_signal(proba, cfg.threshold), index=idx_test, dtype=float)
        sig = sig[~sig.index.duplicated(keep="last")].sort_index()
        sig.index = pd.to_datetime(sig.index).tz_localize(None)
        return sig, idx_test

    def run(
        self,
        ohlcv: pd.DataFrame,
        model: Model,
        tt_cfg: TrainTestConfig,
        ex_cfg: ExecConfig,
    ) -> Tuple[pd.Series, Dict[str, Any], Any]:
        """Execute a backtest run.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            Input OHLCV.
        model : ModelInterface
            Model to evaluate.
        tt_cfg : TrainTestConfig
            Train/test and threshold config.
        ex_cfg : ExecConfig
            Execution parameters.

        Returns
        -------
        Tuple[pd.Series, Dict[str, Any], Any]
            (signal_series_on_test, stats_dict, backtesting_figure)
        """
        feats = self.fe.make(ohlcv)
        if len(feats) < 50:
            raise ValueError("Not enough rows after feature engineering (need >= 50).")

        signal, idx_test = self._build_signal_series(feats, model, tt_cfg)

        bt_df = ohlcv.copy()
        bt_df.index = pd.to_datetime(bt_df.index).tz_localize(None)
        bt_df["signal"] = 0.0
        common = bt_df.index.intersection(signal.index)
        bt_df.loc[common, "signal"] = signal.reindex(common).to_numpy()
        bt_slice = bt_df.loc[idx_test.min() : idx_test.max()].copy()

        class SignalStrategy(Strategy):
            signal_threshold = 0.0
            allow_short = ex_cfg.allow_short

            def init(self_):
                self_.signal = self_.I(lambda: self_.data.df["signal"].values)

            def next(self_):
                s = self_.signal[-1]
                if s > self_.signal_threshold:
                    if self_.position.is_short:
                        self_.position.close()
                    if not self_.position.is_long:
                        self_.buy()
                elif s < -self_.signal_threshold:
                    if self_.position.is_long:
                        self_.position.close()
                    if self_.allow_short and not self_.position.is_short:
                        self_.sell()
                else:
                    if self_.position:
                        self_.position.close()

        bt = Backtest(
            bt_slice[["Open", "High", "Low", "Close", "Volume", "signal"]],
            SignalStrategy,
            cash=ex_cfg.cash,
            commission=ex_cfg.commission,
            exclusive_orders=True,
            finalize_trades=ex_cfg.finalize_trades,
        )

        stats = bt.run()
        fig = bt.plot(open_browser=False)
        logger.info("Backtest complete")
        return signal, dict(stats), fig
