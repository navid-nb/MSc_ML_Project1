"""Console controller for CLI runs (evaluate() only)."""

import argparse
from typing import Any, Dict

import pandas as pd

from backtester.features.engineer import DefaultFeatureEngineer
from backtester.models.buy_hold import BuyHoldModel
from backtester.models.decision_tree import DecisionTreeModel
from backtester.models.gaussian_nb import GaussianNBModel
from backtester.pipelines.evaluate_pipeline import EvalConfig, evaluate
from backtester.providers.adapter_utils import resample_ohlcv
from backtester.providers.csv_asset import CSVAsset
from backtester.providers.yfinance_asset import YFinanceAsset
from backtester.utils.io import read_csv_ohlcv


def _model_factory(name: str):
    name = name.lower()
    if name in ["buy", "buyhold", "buy&hold", "bh"]:
        return BuyHoldModel()
    if name in ["tree", "dt", "decisiontree"]:
        return DecisionTreeModel()
    return GaussianNBModel()


def run_cli(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute an evaluation from CLI arguments and return stats."""
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) if args.end else None

    # Load data
    if args.source == "yfinance":
        asset = YFinanceAsset(args.symbol)
        df = asset.load(start, end, args.freq)
    else:
        raw = read_csv_ohlcv(args.csv)
        asset = CSVAsset("(CSV)", raw)
        df = asset.load(start, end, None)
        if args.freq:
            df = resample_ohlcv(df, args.freq)

    # Evaluate
    model = _model_factory(args.model)
    stats = evaluate(
        ohlcv=df,
        feature_engineer=DefaultFeatureEngineer(),
        model=model,
        cfg=EvalConfig(split_ratio=args.split, cash=args.cash),
    )
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    p = argparse.ArgumentParser(description="Console backtester")
    p.add_argument("--source", choices=["yfinance", "csv"], default="yfinance")
    p.add_argument("--symbol", default="AAPL")
    p.add_argument("--csv", help="Path to CSV (if source=csv)")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--freq", default="1d", help="yfinance interval or resample freq for CSV")
    p.add_argument("--model", default="nb", help="buy | tree | nb")
    p.add_argument("--cash", type=float, default=10000.0)
    p.add_argument("--split", type=float, default=0.7)
    p.add_argument("--save_stats", default=None)
    p.add_argument("--save_equity", default=None)
    return p
