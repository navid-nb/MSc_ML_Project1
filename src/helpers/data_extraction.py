import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import yfinance as yf

from src.helpers._extract import ensure_dir, make_run_folder, safe_delete_dir
from src.helpers._sql import assert_artifacts_present, extract_artifacts, wrds_connect


def wrds_extract_raw(
    wrds_user: str,
    start: str,
    end: str,
    chunk_size: int,
    use_run: str,
    base_dir: str,
    artifacts: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Orchestrate raw data extraction from WRDS with directory and reuse management.

    This function sets up the run folder according to the user's choice (new run, reuse last, or specific run),
    connects to WRDS using given credentials, and runs SQL queries for all requested data artifacts.
    Data is extracted in chunks and saved as Parquet files in the run folder.
    If reuse mode is selected, it validates that the expected Parquet files already exist and skips extraction.

    Args:
        wrds_user (str): WRDS username for authentication.
        start (str): Start date for data extraction (e.g., "YYYY-MM-DD").
        end (str): End date for data extraction (e.g., "YYYY-MM-DD").
        chunk_size (int): Number of rows to fetch per chunk from WRDS.
        use_run (str): Specifies run mode ("new", "last", or explicit run folder name).
        base_dir (str): Base directory under which to organize run folders.
        artifacts (List[Tuple[str, str]]): List of (SQL file path, output Parquet filename) tuples to extract.

    Returns:
        Dict[str, Any]: Contains keys:
            - "run_folder": Path of the run folder used for extraction.
            - "reuse": Boolean flag whether the run folder was reused.
            - "artifacts": Dict mapping Parquet filenames to their absolute paths.
    """
    ensure_dir(base_dir)
    run_dir, run_name, reuse = make_run_folder(base_dir, use_run)
    print(f"[info] Using run folder: {run_name} (reuse={reuse})")

    if not reuse:
        conn = None
        try:
            conn = wrds_connect(wrds_user)
            params = {"start": start, "end": end}
            extract_artifacts(
                conn,
                artifacts,
                run_dir,
                params=params,
                chunk_size=chunk_size,
                force=True,
            )
            print("[info] Extraction complete (full refresh).")
        except Exception:
            safe_delete_dir(run_dir, base_dir)
            raise
        finally:
            if conn is not None:
                conn.close()
    else:
        assert_artifacts_present(run_dir, artifacts)
        print("[info] Reuse mode: all required Parquet files are present. No extraction performed.")

    produced = {
        parq: os.path.join(run_dir, parq)
        for _, parq in artifacts
        if os.path.isfile(os.path.join(run_dir, parq))
    }

    return {
        "run_folder": run_dir,
        "reuse": reuse,
        "artifacts": produced,
    }


def common_features_extract(
    start_date: str, end_date: str, output_path: str = "data/yfinance.parquet"
):
    """
    Download daily data for given tickers from yfinance and save as Parquet.
    Keeps all columns with ticker-prefixed names (e.g., ^vix_Close).
    """
    tickers = [
        # Volatility Indexes
        "^VIX",  # CBOE Volatility Index: 30-day expected volatility of the S&P 500 (market fear gauge)
        "^VXN",  # CBOE NASDAQ-100 Volatility Index: 30-day expected volatility of the Nasdaq-100 (tech-heavy)
        "^OVX",  # CBOE Crude Oil Volatility Index: 30-day expected volatility of WTI Crude Oil futures
        "^GVZ",  # CBOE Gold Volatility Index: 30-day expected volatility of Gold futures
        # Equity Indexes
        "^GSPC",  # S&P 500: U.S. large-cap equity benchmark
        "^IXIC",  # Nasdaq Composite: U.S. tech-heavy index
        "^RUT",  # Russell 2000: U.S. small-cap index
        # Sector ETFs (for sector rotation/flow signals)
        "XLK",  # Technology
        "XLF",  # Financials
        "XLE",  # Energy
        "XLV",  # Health Care
        "XLI",  # Industrials
    ]

    all_data = []

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if data.empty:
            print(f"Warning: No data returned for {ticker}")
            continue

        # Flatten MultiIndex columns: e.g. (Close, ^VIX) -> comm_^VIX_close
        data.columns = [f"comm_{col[1]}_{col[0].lower()}" for col in data.columns]

        data.index.name = "date"
        all_data.append(data)

    # Combine all DataFrames
    if not all_data:
        raise ValueError("No data downloaded for any ticker.")

    df = pd.concat(all_data, axis=1).reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.date  # Ensure date is date type

    # Save to Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"yfinance downloaded data for {tickers}")
    print(f"yfinance data saved to {output_path}")
