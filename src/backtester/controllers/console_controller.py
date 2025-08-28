"""Console controller for CLI runs (multi-asset evaluate())."""

import argparse
from typing import Any, Dict, List

import pandas as pd

from backtester.features.engineer import DefaultFeatureEngineer
from backtester.models.decision_tree import DecisionTreeModel
from backtester.models.gaussian_nb import GaussianNBModel
from backtester.pipelines.evaluate_pipeline import EvalConfig, evaluate
from backtester.providers.yfinance_asset import YFinanceAsset


def _model_factory(name: str):
    name = name.lower()
    if name in ["tree", "dt", "decisiontree"]:
        return DecisionTreeModel()
    return GaussianNBModel()


def _split_symbols(s: str) -> List[str]:
    return [t.strip() for t in s.replace(",", " ").split() if t.strip()]


def _parse_owned(owned: str) -> Dict[str, float]:
    """Parse 'AAPL:10,MSFT:0' string into dict."""
    out: Dict[str, float] = {}
    if not owned:
        return out
    for part in owned.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            out[k.strip()] = float(v.strip())
        except Exception:
            pass
    return out


def run_cli(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute a multi-asset evaluation from CLI arguments and return stats."""
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) if args.end else None

    data_map: Dict[str, pd.DataFrame] = {}
    for sym in _split_symbols(args.symbol):
        asset = YFinanceAsset(sym)
        data_map[sym] = asset.load(start, end, args.freq)

    def model_ctor():
        return _model_factory(args.model)

    # model_ctor = lambda: _model_factory(args.model)
    owned_map = _parse_owned(args.owned)

    stats = evaluate(
        ohlcv=data_map,
        feature_engineer=DefaultFeatureEngineer(),
        model=model_ctor,  # factory for per-asset models
        cfg=EvalConfig(split_ratio=args.split, cash=args.cash),
        initial_shares=owned_map,
    )
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Console backtester (multi-asset)")
    p.add_argument("--source", choices=["yfinance"], default="yfinance")
    p.add_argument("--symbol", default="AAPL,MSFT", help="Comma/space separated list")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--freq", default="1d", help="yfinance interval")
    p.add_argument("--model", default="nb", help="buy | tree | nb")
    p.add_argument("--cash", type=float, default=10000.0)
    p.add_argument("--split", type=float, default=0.7)
    p.add_argument("--owned", default="", help="e.g., 'AAPL:10,MSFT:0'")
    p.add_argument("--save_stats", default=None)
    p.add_argument("--save_equity", default=None)
    return p
