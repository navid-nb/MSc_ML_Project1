import os
import tempfile

import pandas as pd
import quantstats as qs

from functions.helpers._extract import ensure_dir


def _aggregate_simple_returns(returns: pd.Series, freq: str) -> pd.Series:
    """
    Aggregate simple returns to target frequency by compounding:
      monthly: (1+r_d).prod() - 1 per month
    """
    if freq.upper() in ("M", "MS", "ME"):
        return (1.0 + returns).resample("M").prod().sub(1.0).dropna()
    elif freq.upper() in ("D", "B"):
        # already daily; ensure no NA at the start
        return returns.dropna()
    else:
        raise ValueError(f"Unsupported freq: {freq}. Use 'D' or 'M'.")


def upload_html_to_s3(
    html_content: str,
    bucket: str,
    key: str,
    *,
    content_type: str = "text/html",
    s3_client=None,
) -> str:
    """
    Upload an HTML string to S3 and return the s3:// URI.

    Args:
        html_content: The HTML content to upload.
        bucket: Target S3 bucket name.
        key: Target S3 object key (e.g. 'reports/my_report.html').
        content_type: MIME type for the object (default 'text/html').
        s3_client: Optional pre-configured boto3 S3 client. If None, a new one is created.

    Returns:
        str: The S3 URI of the uploaded object (e.g. 's3://bucket/key').

    Raises:
        RuntimeError: If boto3 is not installed.
        botocore.exceptions.ClientError: If the upload fails.
    """
    if s3_client is None:
        try:
            import boto3
        except ImportError as e:
            raise RuntimeError(
                "boto3 is required for S3 uploads but is not installed."
            ) from e
        s3_client = boto3.client("s3")

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=html_content,
        ContentType=content_type,
    )
    s3_uri = f"s3://{bucket}/{key}"
    print(f"    Uploaded report to {s3_uri}")
    return s3_uri


def make_qs_report_from_equity(
    equity_series,
    rf_series,
    mktrf_series,
    title,
    out_path,
    freq: str = "D",
    s3_bucket: str | None = None,
    s3_key: str | None = None,
):
    """
    Generate a QuantStats HTML report from an equity curve.

    Supports two output modes:
      1) Local filesystem (default, when s3_bucket is None):
         - Writes HTML to `out_path` on disk.
      2) S3 mode (when s3_bucket is provided):
         - Renders HTML into a temporary local file (ephemeral storage),
           reads it back, uploads to S3, and deletes the temp file.
         - `s3_key` controls the S3 object key; if omitted, `out_path` is used as key.

    Args:
        equity_series: Equity curve (cumulative returns at DAILY granularity)
        rf_series: Risk-free rate series (daily simple returns)
        mktrf_series: Market excess return series (daily excess returns)
        title: Report title
        out_path: Output file path (local path or used to derive default S3 key)
        freq: 'D' for daily report (default), 'M' for monthly-aggregated report
        s3_bucket: Optional S3 bucket name to upload the HTML report to.
        s3_key: Optional S3 object key; if None and s3_bucket is provided, derives
                a key from `out_path` (stripping leading slashes and normalizing
                backslashes to forward slashes).
    """
    # 1) Daily simple returns from equity
    rets_daily = equity_series.pct_change(fill_method=None).dropna()

    # 2) Align RF & benchmark daily on the same index
    rf_d = rf_series.reindex(rets_daily.index).ffill().bfill()
    mktrf_d = mktrf_series.reindex(rets_daily.index).ffill().bfill()
    bench_daily_simple = (mktrf_d + rf_d).dropna()

    # 3) Aggregate if monthly report is requested
    if freq.upper() == "M":
        rets = _aggregate_simple_returns(rets_daily, "M")
        rf = _aggregate_simple_returns(rf_d, "M")
        bench_simple = _aggregate_simple_returns(bench_daily_simple, "M")
        periods_per_year = 12
        freq_label = "Monthly (aggregated from daily)"
    else:
        rets = rets_daily
        rf = rf_d
        bench_simple = bench_daily_simple
        periods_per_year = 252
        freq_label = "Daily"

    # 4) Excess returns
    strat_excess = (rets - rf).dropna()
    bench_excess = (bench_simple - rf).reindex(strat_excess.index).dropna()

    # 5) Align both on common index
    common_idx = strat_excess.index.intersection(bench_excess.index)
    strat_excess = strat_excess.reindex(common_idx)
    bench_excess = bench_excess.reindex(common_idx)

    # 6) Produce report: local file or S3
    if s3_bucket:
        # Derive default S3 key from out_path if not provided explicitly
        key = s3_key or str(out_path).lstrip("/").replace("\\", "/")

        # Use a temporary local file for quantstats (required `output` argument)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                tmp_path = tmp.name

            qs.reports.html(
                strat_excess,
                benchmark=bench_excess.to_frame("Market"),
                rf=0.0,
                periods_per_year=periods_per_year,
                output=tmp_path,
                title=title,
            )

            # Read back the generated HTML and upload to S3
            with open(tmp_path, "r", encoding="utf-8") as f:
                html_report_content = f.read()

            upload_html_to_s3(html_report_content, bucket=s3_bucket, key=key)
            print(f"   Mode:  S3 upload")
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        print(f"   Freq:  {freq_label}")
        print(f"   Period: {strat_excess.index.min().date()} to {strat_excess.index.max().date()}")
        print(f"   Points: {len(strat_excess)}")

    else:
        # Local filesystem behavior (original)
        qs.reports.html(
            strat_excess,
            benchmark=bench_excess.to_frame("Market"),
            rf=0.0,
            periods_per_year=periods_per_year,
            output=out_path,
            title=title,
        )
        print(f"    Saved: {out_path}")
        print(f"   Mode:  Local file")
        print(f"   Freq:  {freq_label}")
        print(f"   Period: {strat_excess.index.min().date()} to {strat_excess.index.max().date()}")
        print(f"   Points: {len(strat_excess)}")


def generate_oos_report(
    portfolio_result,
    oos_df,
    dates_out_sample,
    output_path="outputs/oos_long_short_tearsheet.html",
    report_title=None,
    output_dir="outputs",
    s3_bucket: str | None = None,
    s3_key: str | None = None,
):
    """
    DAILY report (existing behavior), with optional S3 upload.

    If s3_bucket is None:
        - Writes the report to `output_path` on the local filesystem.
    If s3_bucket is provided:
        - Renders the report into a temp local file, uploads HTML to S3, then deletes the temp file.
    """
    print("Generating Out-Of-Sample HTML Report (Daily)")

    # Ensure local output directory exists only in local mode
    if s3_bucket is None:
        ensure_dir(output_dir)

    # Daily RF & MKTRF averaged across used stocks
    rf_oos = (
        oos_df.loc[:]
        .reset_index()[["date", "rf"]]
        .dropna()
        .groupby("date", as_index=True)["rf"]
        .mean()
        .astype(float)
        .sort_index()
    )
    mktrf_oos = (
        oos_df.loc[:]
        .reset_index()[["date", "mktrf"]]
        .dropna()
        .groupby("date", as_index=True)["mktrf"]
        .mean()
        .astype(float)
        .sort_index()
    )

    if report_title is None:
        start_date = dates_out_sample.min().date()
        end_date = dates_out_sample.max().date()
        report_title = (
            f"OUT-OF-SAMPLE (Daily): Long-Short Market Neutral ({start_date} to {end_date})"
        )

    make_qs_report_from_equity(
        equity_series=portfolio_result["equity"],
        rf_series=rf_oos,
        mktrf_series=mktrf_oos,
        title=report_title,
        out_path=output_path,
        freq="D",
        s3_bucket=s3_bucket,
        s3_key=s3_key,
    )


def generate_oos_report_monthly(
    portfolio_result,
    oos_df,
    dates_out_sample,
    output_path="outputs/oos_long_short_tearsheet_monthly.html",
    report_title=None,
    output_dir="outputs",
    s3_bucket: str | None = None,
    s3_key: str | None = None,
):
    """
    MONTHLY report (aggregates daily returns -> monthly returns by compounding),
    with optional S3 upload.

    If s3_bucket is None:
        - Writes the report to `output_path` on the local filesystem.
    If s3_bucket is provided:
        - Renders the report into a temp local file, uploads HTML to S3, then deletes the temp file.
    """
    print("Generating Out-Of-Sample HTML Report (Monthly Aggregated)")

    # Ensure local output directory exists only in local mode
    if s3_bucket is None:
        ensure_dir(output_dir)

    # Daily RF & MKTRF averaged across used stocks (aggregation to monthly happens inside make_qs_*).
    rf_oos = (
        oos_df.loc[:]
        .reset_index()[["date", "rf"]]
        .dropna()
        .groupby("date", as_index=True)["rf"]
        .mean()
        .astype(float)
        .sort_index()
    )
    mktrf_oos = (
        oos_df.loc[:]
        .reset_index()[["date", "mktrf"]]
        .dropna()
        .groupby("date", as_index=True)["mktrf"]
        .mean()
        .astype(float)
        .sort_index()
    )

    if report_title is None:
        start_date = dates_out_sample.min().date()
        end_date = dates_out_sample.max().date()
        report_title = (
            f"OUT-OF-SAMPLE (Monthly): Long-Short Market Neutral ({start_date} to {end_date})"
        )

    make_qs_report_from_equity(
        equity_series=portfolio_result["equity"],
        rf_series=rf_oos,
        mktrf_series=mktrf_oos,
        title=report_title,
        out_path=output_path,
        freq="M",
        s3_bucket=s3_bucket,
        s3_key=s3_key,
    )
