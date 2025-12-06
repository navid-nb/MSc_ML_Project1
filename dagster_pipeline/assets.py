import datetime
import io
import os

import boto3
import matplotlib
from matplotlib import pyplot as plt

from functions.helpers.output_generation import make_qs_report_from_equity

# Force non-interactive backend for headless environments (Docker/Fargate)
matplotlib.use("Agg")


import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, Config, MetadataValue, Output, asset
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from functions.helpers.allocation_strategies import apply_allocation_strategy
from functions.helpers.data_extraction import wrds_extract_raw
from functions.helpers.portfolio_backtest import (
    calculate_performance_metrics,
    calculate_portfolio_returns,
)
from functions.helpers.split_window import split_rolling_window, split_train_and_test
from run_data import build_model_matrix_from_raw_data


# Configuration Schema
class StrategyConfig(Config):
    start_date: str = "2010-01-01"
    end_date: str = "2025-01-01"
    # Optional: If None, runs purely locally. If set, attempts S3 download/upload.
    s3_bucket: str | None = "stock-trading-app-data"
    input_prefix: str | None = "input_data/recent"
    tickers: list[str] = [
        "AAPL", "ABT", "ACN", "ADBE", "ADP", "AMD", "AMGN", "AMZN",
        "AXP", "BA", "BAC", "BLK", "BSX", "BX", "C", "CAT", "CMCSA",
        "COP", "COST", "CRM", "CSCO", "CVX", "DHR", "DIS", "ETN", "GE",
        "GS", "HD", "HON", "IBM", "INTU", "ISRG", "JNJ", "JPM", "KO",
        "LLY", "LOW", "MA", "MCD", "MRK", "MS", "MSFT", "NFLX", "NVDA",
        "ORCL", "PEP", "PFE", "PG", "PGR", "PM", "QCOM", "RY", "SCHW",
        "SYK", "T", "TJX", "TMO", "TSM", "TXN", "UNH", "UNP", "V", "VZ",
        "WFC", "WMT", "XOM"
    ]
    perform_tuning: bool = False


# ASSET 1: Ingest Data
@asset(compute_kind="s3")
def raw_data_dict(context: AssetExecutionContext, config: StrategyConfig) -> Output[dict]:
    """
    Downloads Parquet files from S3 (or loads locally) and returns a dictionary.
    """
    source_desc = f"S3 bucket: {config.s3_bucket}" if config.s3_bucket else "Local Disk"
    context.log.info(f"Extracting data from {source_desc}, prefix: {config.input_prefix}")

    raw_data = wrds_extract_raw(
        wrds_user="dagster-service",
        start=config.start_date,
        end=config.end_date,
        chunk_size=500_000,
        use_run="last",
        base_dir="data",
        artifacts=[
            ("functions/migrations/001_base_extract.sql", "dsf.parquet"),
            ("functions/migrations/002_ff_factors.sql", "ff.parquet"),
            ("functions/migrations/003_ibes_statsumu.sql", "ibes_stats.parquet"),
            ("functions/migrations/004_ibes_actu.sql", "ibes_act.parquet"),
        ],
        s3_bucket=config.s3_bucket,
        input_prefix=config.input_prefix,
    )

    # Metadata for the UI
    datasets_dict = raw_data["artifacts"]
    sizes = {k: len(v) for k, v in datasets_dict.items() if hasattr(v, "__len__")}

    path_meta = (
        f"s3://{config.s3_bucket}/{config.input_prefix}"
        if config.s3_bucket
        else f"local:data/{config.input_prefix}"
    )

    return Output(
        value=raw_data,
        metadata={
            "datasets": list(datasets_dict.keys()),
            "row_counts": MetadataValue.json(sizes),
            "source_path": path_meta,
        },
    )


# ASSET 2: Feature Engineering
@asset(compute_kind="pandas")
def model_matrix(
    context: AssetExecutionContext, config: StrategyConfig, raw_data_dict: dict
) -> Output[pd.DataFrame]:
    """
    Cleans raw data and builds the feature matrix (Join factors, macro, log returns).
    """
    context.log.info(f"Building model matrix for {len(config.tickers)} tickers")

    df = build_model_matrix_from_raw_data(
        raw_data=raw_data_dict,
        tickers=config.tickers,
    )

    context.log.info(f"Matrix shape: {df.shape}")

    return Output(
        value=df,
        metadata={
            "num_rows": df.shape[0],
            "num_features": df.shape[1],
            "unique_tickers": len(df.index.get_level_values("permno").unique()),
            "date_range": f"{df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}",
        },
    )


# ASSET 3: Train/Test Split Configuration
@asset(compute_kind="python")
def split_config(context: AssetExecutionContext, model_matrix: pd.DataFrame) -> Output[dict]:
    """
    Calculates rolling window sizes and split dates (IS/OOS).
    """
    random_state = 42
    split_pct = 0.65
    ins_dates, dates_out_sample, split_date = split_train_and_test(
        model_matrix, split_pct, random_state
    )

    # Calculate rolling windows
    split_pct_rolling_train = 0.6
    split_pct_rolling_test = 0.2
    target_folds_count = 10

    (
        ins_window_size,
        ins_training_window_size,
        ins_validation_window_size,
        step_size,
        actual_folds,
    ) = split_rolling_window(
        ins_dates,
        split_pct_rolling_train=split_pct_rolling_train,
        split_pct_rolling_test=split_pct_rolling_test,
        target_folds_count=target_folds_count,
    )

    context.log.info(f"Split Date: {split_date}")
    context.log.info(f"Actual Folds: {actual_folds}")

    return Output(
        value={
            "ins_dates": ins_dates,
            "oos_dates": dates_out_sample,
            "split_date": split_date,
            "cv_params": {
                "step_size": step_size,
                "ins_training_window_size": ins_training_window_size,
                "ins_validation_window_size": ins_validation_window_size,
                "actual_folds": actual_folds,
            },
        },
        metadata={
            "split_date": str(split_date),
            "in_sample_days": len(ins_dates),
            "out_sample_days": len(dates_out_sample),
            "cv_folds": actual_folds,
        },
    )


# ASSET 4: Train Models
@asset(compute_kind="sklearn")
def trained_models(
    context: AssetExecutionContext,
    config: StrategyConfig,
    model_matrix: pd.DataFrame,
    split_config: dict,
) -> Output[dict]:
    """
    Trains Logistic (Direction) and Linear (Magnitude) models using In-Sample data.
    Uses custom Rolling Window Cross-Validation if tuning is enabled.
    """
    ins_dates = split_config["ins_dates"]
    cv_params = split_config["cv_params"]

    # Filter In-Sample
    df_ins = model_matrix[model_matrix.index.get_level_values("date").isin(ins_dates)]

    # Target and Features
    DIR_binary = (df_ins["adj_prc_logret_lead1"] > 0).astype(int)
    y_continuous = df_ins["adj_prc_logret_lead1"]

    num_pred_cols = [c for c in df_ins.columns if c not in (["ticker", "adj_prc_logret_lead1"])]
    X_ins = df_ins[num_pred_cols]

    # Reconstruct Custom Rolling CV Splits
    cv_splits = []

    step_size = cv_params["step_size"]
    train_size = cv_params["ins_training_window_size"]
    val_size = cv_params["ins_validation_window_size"]
    actual_folds = cv_params["actual_folds"]

    if config.perform_tuning:
        context.log.info(f"Constructing {actual_folds} Rolling CV folds...")
        for fold_idx in range(actual_folds):
            start_idx = fold_idx * step_size
            train_end_idx = start_idx + train_size
            val_end_idx = train_end_idx + val_size

            # Get the specific dates for this fold
            fold_train_dates = ins_dates[start_idx:train_end_idx]
            fold_val_dates = ins_dates[train_end_idx:val_end_idx]

            # Find matching integer indices in the DataFrame
            train_mask = df_ins.index.get_level_values("date").isin(fold_train_dates)
            val_mask = df_ins.index.get_level_values("date").isin(fold_val_dates)

            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]

            if len(train_indices) > 0 and len(val_indices) > 0:
                cv_splits.append((train_indices, val_indices))

        context.log.info(f"Generated {len(cv_splits)} valid CV splits.")

    # 1. Logistic Regression
    context.log.info("Training Logistic Regression")

    l1_ratio_log = 0.9
    C_log = 0.1

    if config.perform_tuning:
        context.log.info("Performing Grid Search for Logistic Regression...")
        base_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        max_iter=5000,
                        tol=1e-4,
                        random_state=42,
                    ),
                ),
            ]
        )

        param_grid = {"logistic__l1_ratio": [0.8, 0.9], "logistic__C": [0.1, 0.5, 1.0]}

        grid = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv_splits,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_ins, DIR_binary)

        l1_ratio_log = grid.best_params_["logistic__l1_ratio"]
        C_log = grid.best_params_["logistic__C"]
        context.log.info(
            f"Optimal Logistic: l1={l1_ratio_log}, C={C_log}, Acc={grid.best_score_:.4f}"
        )
    else:
        context.log.info(f"Skipping tuning. Using hardcoded l1={l1_ratio_log}, C={C_log}")

    final_pipeline_log = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    penalty="elasticnet",
                    C=C_log,
                    l1_ratio=l1_ratio_log,
                    solver="saga",
                    max_iter=5000,
                    tol=1e-4,
                    random_state=42,
                ),
            ),
        ]
    )
    final_pipeline_log.fit(X_ins, DIR_binary)

    # Top Features Log
    coefs_log = final_pipeline_log.named_steps["logistic"].coef_[0]
    coef_df = pd.DataFrame({"feature": num_pred_cols, "coef": coefs_log})
    top_features = coef_df.reindex(coef_df.coef.abs().sort_values(ascending=False).index).head(10)
    context.log.info(f"Top 10 Logistic Predictors:\n{top_features.to_string(index=False)}")

    # 2. Linear Regression
    context.log.info("Training Linear Regression")

    l1_ratio_lin = 0.5
    alpha_lin = 0.0001

    if config.perform_tuning:
        context.log.info("Performing Grid Search for Linear Regression...")
        base_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lasso", ElasticNet(max_iter=100000, tol=1e-4, random_state=42)),
            ]
        )

        param_grid = {
            "lasso__l1_ratio": [0.4, 0.5, 0.6],
            "lasso__alpha": [0.000005, 0.00005, 0.0005, 0.000001, 0.00001, 0.0001],
        }

        grid = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv_splits,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_ins, y_continuous)

        l1_ratio_lin = grid.best_params_["lasso__l1_ratio"]
        alpha_lin = grid.best_params_["lasso__alpha"]
        min_rmse = -grid.best_score_
        context.log.info(
            f"Optimal Linear: l1={l1_ratio_lin}, alpha={alpha_lin}, RMSE={min_rmse:.6f}"
        )
    else:
        context.log.info(f"Skipping tuning. Using hardcoded l1={l1_ratio_lin}, alpha={alpha_lin}")

    final_pipeline_lin = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lasso",
                ElasticNet(
                    alpha=alpha_lin,
                    l1_ratio=l1_ratio_lin,
                    max_iter=100000,
                    tol=1e-4,
                    random_state=42,
                ),
            ),
        ]
    )
    final_pipeline_lin.fit(X_ins, y_continuous)

    return Output(
        value={
            "logistic_pipeline": final_pipeline_log,
            "linear_pipeline": final_pipeline_lin,
            "features": num_pred_cols,
        },
        metadata={
            "logistic_l1_ratio": l1_ratio_log,
            "logistic_C": C_log,
            "linear_l1_ratio": l1_ratio_lin,
            "linear_alpha": alpha_lin,
            "top_predictor": top_features.iloc[0]["feature"],
            "features_count": len(num_pred_cols),
        },
    )


# ASSET 5: Generate Signals (Full History)
@asset(compute_kind="pandas")
def trading_signals(
    context: AssetExecutionContext, model_matrix: pd.DataFrame, trained_models: dict
) -> Output[pd.DataFrame]:
    """
    Applies models to full dataset. Calculates 'Ensemble Agreement'.
    """
    pipeline_log = trained_models["logistic_pipeline"]
    pipeline_lin = trained_models["linear_pipeline"]
    features = trained_models["features"]

    X_full = model_matrix[features]

    # Predictions
    prob_up = pipeline_log.predict_proba(X_full)[:, 1]
    expected_return = pipeline_lin.predict(X_full)

    df_signals = model_matrix.copy()
    df_signals["prob_up"] = prob_up
    df_signals["expected_return"] = expected_return
    df_signals["logistic_score"] = 2 * prob_up - 1

    # Thresholds
    PROB_UP_THRESHOLD = 0.55
    EXPECTED_RETURN_THRESHOLD = 0.001

    df_signals["logistic_signal_long"] = df_signals["prob_up"] > PROB_UP_THRESHOLD
    df_signals["logistic_signal_short"] = df_signals["prob_up"] < (1 - PROB_UP_THRESHOLD)

    df_signals["linear_signal_long"] = df_signals["expected_return"] > EXPECTED_RETURN_THRESHOLD
    df_signals["linear_signal_short"] = df_signals["expected_return"] < -EXPECTED_RETURN_THRESHOLD

    # Agreement
    df_signals["agreed_long"] = (
        df_signals["logistic_signal_long"] & df_signals["linear_signal_long"]
    )
    df_signals["agreed_short"] = (
        df_signals["logistic_signal_short"] & df_signals["linear_signal_short"]
    )
    df_signals["agreed_any"] = df_signals["agreed_long"] | df_signals["agreed_short"]
    df_signals["disagreed"] = ~df_signals["agreed_any"]

    # Agreement Statistics Logs
    total_obs = len(df_signals)
    agree_long_pct = df_signals["agreed_long"].mean() * 100
    agree_short_pct = df_signals["agreed_short"].mean() * 100
    disagree_pct = df_signals["disagreed"].mean() * 100

    context.log.info(
        f"Agreement Stats: Long {agree_long_pct:.1f}%, Short {agree_short_pct:.1f}%, Disagree {disagree_pct:.1f}%"
    )

    return Output(
        value=df_signals,
        metadata={
            "total_observations": float(total_obs),
            "agreement_long_pct": float(agree_long_pct),
            "agreement_short_pct": float(agree_short_pct),
            "disagreement_pct": float(disagree_pct),
            "preview": MetadataValue.md(
                df_signals[["prob_up", "expected_return", "agreed_long"]].head().to_markdown()
            ),
        },
    )


# ASSET 6a: In-Sample Strategy Selection
@asset(compute_kind="python")
def best_strategy_config(
    context: AssetExecutionContext,
    trading_signals: pd.DataFrame,
    split_config: dict,
) -> Output[dict]:
    """
    Iterates through all scoring (S1..S6) and allocation (A1..A5) combinations
    on IN-SAMPLE data to find the configuration with the highest Sharpe Ratio.
    """
    ins_dates = split_config["ins_dates"]

    context.log.info("Starting In-Sample Strategy Selection")

    # Filter to In-Sample
    df_ins = trading_signals[trading_signals.index.get_level_values("date").isin(ins_dates)].copy()

    # Pre-calculate Volatility (Needed for A5 strategy)
    if "anualized_volatility_20d" in df_ins.columns:
        df_ins["volatility"] = df_ins["anualized_volatility_20d"] / np.sqrt(252)
    else:
        df_ins["volatility"] = df_ins.groupby(level="permno")["adj_prc_logret"].transform(
            lambda x: x.rolling(window=20, min_periods=5).std()
        )
    df_ins["volatility"] = df_ins.groupby(level="date")["volatility"].transform(
        lambda x: x.fillna(x.median())
    )

    # Define Scoring Methods
    SCORING_METHODS = {
        "S1": ("score_S1", lambda df: df["prob_up"] * df["expected_return"]),
        "S2": ("score_S2", lambda df: (df["prob_up"] - 0.5) * df["expected_return"]),
        "S6": ("score_S6", lambda df: (2 * df["prob_up"] - 1) * df["expected_return"].abs()),
    }

    ALLOCATION_STRATEGIES = ["A1", "A2", "A3", "A4", "A5"]

    # Calculate Scores Columns
    for name, (col, func) in SCORING_METHODS.items():
        df_ins[col] = func(df_ins)

    results = []

    # Constants
    LONG_TARGET = 1.0
    SHORT_TARGET = 1.0
    MAX_POSITION_SIZE = 0.05
    QUANTILE_LONG_PCT = 0.20
    QUANTILE_SHORT_PCT = 0.20

    total_combos = len(SCORING_METHODS) * len(ALLOCATION_STRATEGIES)
    context.log.info(f"Testing {total_combos} combinations on In-Sample data...")

    for score_name, (score_col, score_desc) in SCORING_METHODS.items():
        for alloc_strategy in ALLOCATION_STRATEGIES:
            scores = df_ins[score_col].copy()
            long_mask = df_ins["agreed_long"]
            short_mask = df_ins["agreed_short"]

            df_ins_work = df_ins.copy()
            weights_all = []

            dates = df_ins_work.index.get_level_values("date").unique()

            for date in dates:
                date_mask = df_ins_work.index.get_level_values("date") == date

                alloc_params = {
                    "long_target": LONG_TARGET,
                    "short_target": SHORT_TARGET,
                    "max_position_size": MAX_POSITION_SIZE,
                }

                if alloc_strategy == "A3":
                    alloc_params["quantile_long_pct"] = QUANTILE_LONG_PCT
                    alloc_params["quantile_short_pct"] = QUANTILE_SHORT_PCT

                kwargs = {}
                if alloc_strategy == "A5":
                    kwargs["volatility"] = df_ins_work.loc[date_mask, "volatility"]

                weights_date = apply_allocation_strategy(
                    strategy_name=alloc_strategy,
                    scores=scores[date_mask],
                    long_mask=long_mask[date_mask],
                    short_mask=short_mask[date_mask],
                    **kwargs,
                    **alloc_params,
                )
                weights_all.append(weights_date)

            weights_series = pd.concat(weights_all)
            df_ins_work["portfolio_weights"] = weights_series

            portfolio_returns = calculate_portfolio_returns(
                df=df_ins_work,
                weights_col="portfolio_weights",
                returns_col="adj_prc_logret_lead1",
                date_col="date",
            )

            metrics = calculate_performance_metrics(
                returns=portfolio_returns, rf_rate=0.0, periods_per_year=252
            )

            results.append(
                {
                    "scoring_method": score_name,
                    "allocation_strategy": alloc_strategy,
                    "sharpe": metrics["sharpe"],
                    "ann_return": metrics["ann_return"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise Exception("No strategies successfully completed In-Sample testing.")

    # Select Best
    best_combo = results_df.loc[results_df["sharpe"].idxmax()]
    best_scoring = best_combo["scoring_method"]
    best_alloc = best_combo["allocation_strategy"]
    best_is_sharpe = best_combo["sharpe"]

    context.log.info(f"Optimal In-Sample Strategy: {best_scoring} + {best_alloc}")
    context.log.info(f"IS Sharpe: {best_is_sharpe:.3f}, Return: {best_combo['ann_return']:.2f}%")

    return Output(
        value={
            "scoring_method": best_scoring,
            "allocation_strategy": best_alloc,
            "is_sharpe": best_is_sharpe,
            "is_metrics": best_combo.to_dict(),
        },
        metadata={
            "selected_strategy": f"{best_scoring} + {best_alloc}",
            "is_sharpe": float(best_is_sharpe),
            "top_3_candidates": MetadataValue.md(
                results_df.nlargest(3, "sharpe")[
                    ["scoring_method", "allocation_strategy", "sharpe"]
                ].to_markdown()
            ),
        },
    )


# ASSET 6b: Out-of-Sample Reporting
@asset(compute_kind="quantstats")
def oos_report(
        context: AssetExecutionContext,
        config: StrategyConfig,
        trading_signals: pd.DataFrame,
        split_config: dict,
        best_strategy_config: dict,
) -> Output[dict]:
    """
    1. Retrieves best strategy config from In-Sample selection.
    2. Runs that strategy on Out-of-Sample data.
    3. Generates HTML report.
    """
    # Retrieve inputs
    oos_dates = split_config["oos_dates"]

    best_scoring = best_strategy_config["scoring_method"]
    best_alloc = best_strategy_config["allocation_strategy"]
    best_is_sharpe = best_strategy_config["is_sharpe"]

    context.log.info(
        f"Running Optimal Strategy ({best_scoring}+{best_alloc}) on Out-of-Sample"
    )

    # Filter to OOS
    df_oos = trading_signals[trading_signals.index.get_level_values("date").isin(oos_dates)].copy()

    # Calculate OOS Volatility (Needed for A5)
    if "anualized_volatility_20d" in df_oos.columns:
        df_oos["volatility"] = df_oos["anualized_volatility_20d"] / np.sqrt(252)
    else:
        df_oos["volatility"] = df_oos.groupby(level="permno")["adj_prc_logret"].transform(
            lambda x: x.rolling(window=20, min_periods=5).std()
        )
    df_oos["volatility"] = df_oos.groupby(level="date")["volatility"].transform(
        lambda x: x.fillna(x.median())
    )

    # Re-define Scoring (Logic must match previous asset)
    SCORING_METHODS = {
        "S1": ("score_S1", lambda df: df["prob_up"] * df["expected_return"]),
        "S2": ("score_S2", lambda df: (df["prob_up"] - 0.5) * df["expected_return"]),
        "S6": ("score_S6", lambda df: (2 * df["prob_up"] - 1) * df["expected_return"].abs()),
    }

    # Apply Selected Score
    score_col_oos = SCORING_METHODS[best_scoring][0]
    func_oos = SCORING_METHODS[best_scoring][1]
    df_oos[score_col_oos] = func_oos(df_oos)

    # Apply Selected Allocation
    weights_all_oos = []
    unique_dates_oos = df_oos.index.get_level_values("date").unique()
    scores_oos = df_oos[score_col_oos]
    long_mask_oos = df_oos["agreed_long"]
    short_mask_oos = df_oos["agreed_short"]

    # Constants
    LONG_TARGET = 1.0
    SHORT_TARGET = 1.0
    MAX_POSITION_SIZE = 0.05
    QUANTILE_LONG_PCT = 0.20
    QUANTILE_SHORT_PCT = 0.20

    for date in unique_dates_oos:
        date_mask = df_oos.index.get_level_values("date") == date

        alloc_params = {
            "long_target": LONG_TARGET,
            "short_target": SHORT_TARGET,
            "max_position_size": MAX_POSITION_SIZE,
        }

        if best_alloc == "A3":
            alloc_params["quantile_long_pct"] = QUANTILE_LONG_PCT
            alloc_params["quantile_short_pct"] = QUANTILE_SHORT_PCT

        kwargs = {}
        if best_alloc == "A5":
            kwargs["volatility"] = df_oos.loc[date_mask, "volatility"]

        weights_date = apply_allocation_strategy(
            strategy_name=best_alloc,
            scores=scores_oos[date_mask],
            long_mask=long_mask_oos[date_mask],
            short_mask=short_mask_oos[date_mask],
            **kwargs,
            **alloc_params,
        )
        weights_all_oos.append(weights_date)

    df_oos["portfolio_weights"] = pd.concat(weights_all_oos)

    # Calculate Returns
    portfolio_returns = calculate_portfolio_returns(
        df=df_oos,
        weights_col="portfolio_weights",
        returns_col="adj_prc_logret_lead1",
        date_col="date",
    )

    metrics = calculate_performance_metrics(portfolio_returns, rf_rate=0.0)
    context.log.info(f"OOS Sharpe: {metrics['sharpe']:.3f}")

    degradation = (metrics["sharpe"] - best_is_sharpe) / best_is_sharpe * 100
    context.log.info(f"Sharpe Degradation: {degradation:.1f}%")

    # Generate Report
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    equity_curve = (1 + portfolio_returns).cumprod()

    rf_series = df_oos.reset_index().groupby("date")["rf"].mean()
    mktrf_series = df_oos.reset_index().groupby("date")["mktrf"].mean()

    report_path = f"outputs/run_{now}/oos_{best_scoring}_{best_alloc}.html"

    make_qs_report_from_equity(
        equity_series=equity_curve,
        rf_series=rf_series,
        mktrf_series=mktrf_series,
        title=f"OOS: {best_scoring}+{best_alloc}",
        out_path=report_path,
        freq="D",
        s3_bucket=config.s3_bucket,
    )

    # Detect Report URL (Local vs S3)
    if config.s3_bucket:
        report_url = (
            f"https://s3.console.aws.amazon.com/s3/object/{config.s3_bucket}?prefix={report_path}"
        )
    else:
        abs_path = os.path.abspath(report_path)
        report_url = f"file://{abs_path}"

    return Output(
        value={
            "status": "success",
            "report_path": report_path,
            "final_equity": equity_curve.iloc[-1],
        },
        metadata={
            "selected_strategy": f"{best_scoring} + {best_alloc}",
            "is_sharpe": float(best_is_sharpe),
            "oos_sharpe": float(metrics["sharpe"]),
            "degradation_pct": float(degradation),
            "oos_cagr_pct": float(metrics["ann_return"]),
            "oos_max_dd_pct": float(metrics["max_drawdown"]),
            "report_url": MetadataValue.url(report_url),
        },
    )


# ASSET 7: Next Day Inference
@asset(compute_kind="python")
def next_day_inference(
        context: AssetExecutionContext,
        config: StrategyConfig,
        model_matrix: pd.DataFrame,
        trained_models: dict,
        best_strategy_config: dict,
        oos_report: dict,
) -> Output[pd.DataFrame]:
    """
    PREDICTION ENGINE:
    1. Takes the latest available data point (Today's Close).
    2. Predicts Tomorrow's Return and Direction.
    3. Generates a 'Buy/Sell/Hold' list based on the Optimal Strategy.
    4. Calculates Best/Worst case portfolio scenarios.
    """
    # 1. Get Latest Data Snapshot
    last_date = model_matrix.index.get_level_values("date").max()
    context.log.info(f"Generating inference for trading date following: {last_date}")

    latest_df = model_matrix[model_matrix.index.get_level_values("date") == last_date].copy()

    # 2. Run Inference
    pipeline_log = trained_models["logistic_pipeline"]
    pipeline_lin = trained_models["linear_pipeline"]
    features = trained_models["features"]

    X_latest = latest_df[features]

    # Predict Probabilities and Returns
    prob_up = pipeline_log.predict_proba(X_latest)[:, 1]
    expected_ret = pipeline_lin.predict(X_latest)

    latest_df["prob_up"] = prob_up
    latest_df["expected_return"] = expected_ret
    latest_df["logistic_score"] = 2 * prob_up - 1

    # LOGGING: Debug why we might have 0 trades
    context.log.info(f"Prediction Stats -- Max Prob: {prob_up.max():.4f}, Min Prob: {prob_up.min():.4f}")
    context.log.info(f"Return Stats -- Max Ret: {expected_ret.max():.4f}, Min Ret: {expected_ret.min():.4f}")

    # 3. Apply The "Winning" Strategy Logic
    best_scoring = best_strategy_config["scoring_method"]
    best_alloc = best_strategy_config["allocation_strategy"]

    context.log.info(f"Applying Winning Strategy: {best_scoring} + {best_alloc}")

    SCORING_METHODS = {
        "S1": lambda df: df["prob_up"] * df["expected_return"],
        "S2": lambda df: (df["prob_up"] - 0.5) * df["expected_return"],
        "S6": lambda df: (2 * df["prob_up"] - 1) * df["expected_return"].abs(),
    }

    latest_df["score"] = SCORING_METHODS[best_scoring](latest_df)

    PROB_UP_THRESHOLD = 0.55
    RET_THRESHOLD = 0.001

    latest_df["agreed_long"] = (latest_df["prob_up"] > PROB_UP_THRESHOLD) & (
            latest_df["expected_return"] > RET_THRESHOLD
    )
    latest_df["agreed_short"] = (latest_df["prob_up"] < (1 - PROB_UP_THRESHOLD)) & (
            latest_df["expected_return"] < -RET_THRESHOLD
    )

    # Volatility Calculation
    vol_series = (
        model_matrix.loc[pd.IndexSlice[:, last_date], :]
        .groupby("permno")["adj_prc_logret"]
        .apply(lambda x: x.rolling(20).std().iloc[-1])
    )
    latest_df["volatility"] = vol_series

    # Apply Allocation
    alloc_kwargs = {
        "long_target": 1.0,
        "short_target": 1.0,
        "max_position_size": 0.05,
        "quantile_long_pct": 0.20,
        "quantile_short_pct": 0.20,
    }

    weights = apply_allocation_strategy(
        strategy_name=best_alloc,
        scores=latest_df["score"],
        long_mask=latest_df["agreed_long"],
        short_mask=latest_df["agreed_short"],
        volatility=latest_df["volatility"],
        **alloc_kwargs,
    )

    latest_df["weight"] = weights

    # 4. Filter for Actionable Trades
    trades = latest_df[latest_df["weight"] != 0].copy()

    # Handle the case where trades might be empty
    if not trades.empty:
        trades["action"] = trades["weight"].apply(lambda x: "BUY" if x > 0 else "SELL/SHORT")
        trades["conviction"] = trades["score"].abs()
        display_cols = ["prob_up", "expected_return", "weight", "action", "conviction"]
        trade_list = trades[display_cols].sort_values("weight", ascending=False)

        # 5. Risk / Scenario Analysis
        port_exp_return = (trades["weight"] * trades["expected_return"]).sum()
        avg_volatility = 0.015
        port_vol_est = np.sqrt((trades["weight"] ** 2).sum()) * avg_volatility
        worst_case = port_exp_return - 1.96 * port_vol_est
        best_case = port_exp_return + 1.96 * port_vol_est
    else:
        # Fallback for 0 trades
        context.log.warning("No trades generated for this date. Portfolio is 100% Cash.")
        trade_list = pd.DataFrame(columns=["prob_up", "expected_return", "weight", "action", "conviction"])
        port_exp_return = 0.0
        worst_case = 0.0
        best_case = 0.0

    # 6. Visualization & Saving (In-Memory)

    # Helper to save plot to RAM and upload/write
    def save_plot_to_destination(filename_suffix):
        # Derive path from previous report path
        previous_report_path = oos_report["report_path"]
        base_dir = os.path.dirname(previous_report_path)

        # 1. Save figure to in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()  # Close plot to free memory

        # 2. Upload or Write
        if config.s3_bucket:
            s3 = boto3.client("s3")
            s3_key = f"{base_dir}/{filename_suffix}"
            context.log.info(f"Uploading {filename_suffix} to S3: {s3_key}")
            s3.upload_fileobj(buf, config.s3_bucket, s3_key)
            return f"https://s3.console.aws.amazon.com/s3/object/{config.s3_bucket}?prefix={s3_key}"
        else:
            # Local write
            full_path = f"{base_dir}/{filename_suffix}"
            os.makedirs(base_dir, exist_ok=True)
            with open(full_path, "wb") as f:
                f.write(buf.getbuffer())
            context.log.info(f"Saved {filename_suffix} locally: {full_path}")
            return f"file://{os.path.abspath(full_path)}"

    # Plot 1: Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(
        latest_df["expected_return"] * 100,
        bins=30,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    plt.axvline(0, color="k", linestyle="--")
    plt.title(f"Distribution of Predicted Returns for {last_date} (Universe)")
    plt.xlabel("Predicted Return (%)")
    plt.ylabel("Count of Tickers")

    dist_url = save_plot_to_destination("forecast_distribution.png")

    # Plot 2: Scenarios
    plt.figure(figsize=(8, 6))
    scenarios = ["Worst Case (95%)", "Expected", "Best Case (95%)"]
    vals = [worst_case * 100, port_exp_return * 100, best_case * 100]
    colors = ["red", "blue", "green"]
    plt.bar(scenarios, vals, color=colors, alpha=0.7)
    plt.title(f"Portfolio Forecast for Tomorrow")
    plt.ylabel("Return (%)")
    plt.grid(axis="y", alpha=0.3)

    scenarios_url = save_plot_to_destination("forecast_scenarios.png")

    return Output(
        value=trade_list,
        metadata={
            "inference_date": str(last_date),
            "portfolio_expected_return": f"{port_exp_return * 100:.2f}%",
            "worst_case_95pct": f"{worst_case * 100:.2f}%",
            "best_case_95pct": f"{best_case * 100:.2f}%",
            "num_positions": len(trade_list),
            "distribution_plot": MetadataValue.url(dist_url),
            "scenarios_plot": MetadataValue.url(scenarios_url),
            "top_longs": MetadataValue.md(
                trade_list[trade_list["weight"] > 0].head(5).to_markdown() if not trade_list.empty else "No Longs"
            ),
            "full_trade_list": MetadataValue.md(
                trade_list.to_markdown() if not trade_list.empty else "No Trades Generated"),
        },
    )
