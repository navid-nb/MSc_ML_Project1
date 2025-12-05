import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import yfinance as yf

from functions.helpers._extract import ensure_dir, make_run_folder, safe_delete_dir
from functions.helpers._sql import (
    assert_artifacts_present,
    extract_artifacts,
    wrds_connect,
)


def wrds_extract_raw(
    wrds_user: str,
    start: str,
    end: str,
    chunk_size: int,
    use_run: str,
    base_dir: str,
    artifacts: List[Tuple[str, str]],
    s3_bucket: str | None = None,
    input_prefix: str | None = None,
) -> Dict[str, Any]:
    """
    Orchestrate raw data extraction from WRDS with directory and reuse management.

    This function can operate in two modes:

    1) Local mode (default, no s3_bucket/input_prefix):
       - Uses local run folders under `base_dir`.
       - Extracts data from WRDS (or reuses the last run) into Parquet files.

    2) S3 "read-only" mode (when s3_bucket and input_prefix are provided):
       - Skips WRDS extraction entirely.
       - Assumes the Parquet files already exist in S3 under:
         s3://{s3_bucket}/{input_prefix}/{parquet_name}
       - Returns a `raw_data` dict whose artifact paths are S3 URIs.

    Args:
        wrds_user (str): WRDS username for authentication.
        start (str): Start date for data extraction (e.g., "YYYY-MM-DD").
        end (str): End date for data extraction (e.g., "YYYY-MM-DD").
        chunk_size (int): Number of rows to fetch per chunk from WRDS.
        use_run (str): Specifies run mode ("new", "last", or explicit run folder name).
        base_dir (str): Base directory under which to organize run folders (local mode).
        artifacts (List[Tuple[str, str]]): List of (SQL file path, output Parquet filename).
        s3_bucket (str | None): If provided with input_prefix, enables S3 read-only mode.
        input_prefix (str | None): S3 prefix (folder) where artifacts live.

    Returns:
        Dict[str, Any]: Contains keys:
            - "run_folder": Path of the run folder (local path or S3 URI).
            - "reuse": Boolean flag whether the run folder was reused.
            - "artifacts": Dict mapping Parquet filenames to their absolute paths
                           (local or s3://).
    """
    if s3_bucket and input_prefix:
        prefix_clean = input_prefix.strip("/")
        run_dir = f"s3://{s3_bucket}/{prefix_clean}" if prefix_clean else f"s3://{s3_bucket}"
        print(f"[info] Using S3 artifacts from: {run_dir}")

        produced = {
            parq: f"{run_dir}/{parq}"
            for _, parq in artifacts
        }

        return {
            "run_folder": run_dir,
            "reuse": True,
            "artifacts": produced,
        }

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

    if use_run == "new":
        common_features_extract(
            start_date=start,
            end_date=end,
            output_path=os.path.join(run_dir, "common_features.parquet"),
        )
    else:
        assert os.path.isfile(os.path.join(run_dir, "common_features.parquet"))

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
