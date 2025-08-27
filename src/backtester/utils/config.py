"""Config loaders (YAML + pydantic)."""

import pathlib
from dataclasses import dataclass
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML into dict."""
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class AppConfig:
    """Aggregate configuration."""

    streamlit_title: str
    streamlit_layout: str
    data: Dict[str, Any]
    backtest: Dict[str, Any]
    models: Dict[str, Any]
    logging: Dict[str, Any]

    @classmethod
    def from_dirs(cls, base_dir: str = "configs") -> "AppConfig":
        app_cfg = load_yaml(f"{base_dir}/app.yaml")
        data_cfg = load_yaml(f"{base_dir}/data.yaml")
        backtest_cfg = load_yaml(f"{base_dir}/backtest.yaml")
        logging_cfg = load_yaml(f"{base_dir}/logging.yaml")
        models_cfg = {
            "decision_tree": load_yaml(f"{base_dir}/models/decision_tree.yaml"),
            "gaussian_nb": load_yaml(f"{base_dir}/models/gaussian_nb.yaml"),
        }
        return cls(
            streamlit_title=app_cfg.get("streamlit", {}).get("page_title", "Backtester"),
            streamlit_layout=app_cfg.get("streamlit", {}).get("layout", "wide"),
            data=data_cfg,
            backtest=backtest_cfg,
            models=models_cfg,
            logging=logging_cfg,
        )
