from dagster import (
    Definitions,
    load_assets_from_modules,
    define_asset_job,
    ScheduleDefinition,
    AssetSelection,
)

from . import assets

all_assets = load_assets_from_modules([assets])

full_backtest_job = define_asset_job(
    name="full_strategy_backtest_job",
    selection=AssetSelection.all(),
    description="Run Strategy for 66 most performing stocks with fine tuning disabled.",
)

tech_universe_job = define_asset_job(
    name="tech_sector_backtest_job",
    selection=AssetSelection.all(),
    description="Run Strategy for 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMD' with fine tuning enabled",
    config={
        "ops": {
            "raw_data_dict": {
                "config": {
                    "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMD"],
                    "s3_bucket": "stock-trading-app-data",
                    "input_prefix": "input_data/run_20251012_120713",
                    "perform_tuning": True,
                }
            },
            "model_matrix": {
                "config": {
                     "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMD"]
                }
            }
        }
    }
)

daily_schedule = ScheduleDefinition(
    job=tech_universe_job,
    cron_schedule="0 8 * * *",  # 8:00 AM daily
    name="daily_trading_schedule",
)

# Export all
defs = Definitions(
    assets=all_assets,
    jobs=[full_backtest_job, tech_universe_job],
    schedules=[daily_schedule],
)