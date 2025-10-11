import quantstats as qs

from src.helpers._extract import ensure_dir


def make_qs_report_from_equity(equity_series, rf_series, mktrf_series, title, out_path):
    """
    Generate a QuantStats HTML report from an equity curve.

    Args:
        equity_series: Equity curve (cumulative returns)
        rf_series: Risk-free rate series
        mktrf_series: Market excess return series
        title: Report title
        out_path: Output file path
    """
    # Daily simple returns from equity
    rets = equity_series.pct_change(fill_method=None).dropna()
    # Align RF & benchmark on the same index
    rf_aligned = rf_series.reindex(rets.index).ffill().bfill()
    mktrf_aln = mktrf_series.reindex(rets.index).ffill().bfill()
    bench_simple = mktrf_aln + rf_aligned
    # Excess returns
    strat_excess = (rets - rf_aligned).dropna()
    bench_excess = (bench_simple - rf_aligned).reindex(strat_excess.index).dropna()
    # Align both
    common_idx = strat_excess.index.intersection(bench_excess.index)
    strat_excess = strat_excess.reindex(common_idx)
    bench_excess = bench_excess.reindex(common_idx)

    qs.reports.html(
        strat_excess,
        benchmark=bench_excess.to_frame("Market"),
        rf=0.0,
        periods_per_year=252,
        output=out_path,
        title=title,
    )
    print(f"    Saved: {out_path}")
    print(f"   Period: {strat_excess.index.min().date()} to {strat_excess.index.max().date()}")
    print(f"   Days:   {len(strat_excess)}")


def generate_oos_report(
    portfolio_result,
    oos_df,
    dates_out_sample,
    output_path="out/oos_long_short_tearsheet.html",
    report_title=None,
    output_dir="out",
):
    """
    Generate an out-of-sample HTML performance report.

    This function prepares Fama-French benchmark data and generates a comprehensive
    QuantStats HTML report comparing the portfolio strategy to the market.

    Args:
        portfolio_result: Dictionary with portfolio results containing:
            - "weights": Series of portfolio weights
            - "equity": Series of equity curve (cumulative returns)
        oos_df: Out-of-sample DataFrame with columns:
            - MultiIndex: (permno, date)
            - "rf": Risk-free rate
            - "mktrf": Market excess return
        dates_out_sample: DatetimeIndex of out-of-sample dates
        output_path: Path for the output HTML file (default: "out/oos_long_short_tearsheet.html")
        report_title: Custom report title (optional, auto-generated if None)
        output_dir: Directory to create if it doesn't exist (default: "out")

    Returns:
        None (generates HTML file)
    """
    print("Generating Out-Of-Sample HTML Report")

    # Ensure output directory exists
    ensure_dir(output_dir)

    # Prepare OOS Fama-French series
    # Extract dates where positions exist
    used_mask_oos = portfolio_result["weights"] != 0

    # Calculate average rf and mktrf across stocks for each date
    rf_oos = (
        oos_df.loc[used_mask_oos]
        .reset_index()[["date", "rf"]]
        .dropna()
        .groupby("date", as_index=True)["rf"]
        .mean()
        .astype(float)
        .sort_index()
    )
    mktrf_oos = (
        oos_df.loc[used_mask_oos]
        .reset_index()[["date", "mktrf"]]
        .dropna()
        .groupby("date", as_index=True)["mktrf"]
        .mean()
        .astype(float)
        .sort_index()
    )

    # Auto-generate title if not provided
    if report_title is None:
        start_date = dates_out_sample.min().date()
        end_date = dates_out_sample.max().date()
        report_title = f"OUT-OF-SAMPLE: Long-Short Market Neutral ({start_date} to {end_date})"

    # Generate the report
    make_qs_report_from_equity(
        equity_series=portfolio_result["equity"],
        rf_series=rf_oos,
        mktrf_series=mktrf_oos,
        title=report_title,
        out_path=output_path,
    )
