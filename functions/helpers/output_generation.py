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


def make_qs_report_from_equity(
    equity_series,
    rf_series,
    mktrf_series,
    title,
    out_path,
    freq: str = "D",
):
    """
    Generate a QuantStats HTML report from an equity curve.

    Args:
        equity_series: Equity curve (cumulative returns at DAILY granularity)
        rf_series: Risk-free rate series (daily simple returns)
        mktrf_series: Market excess return series (daily excess returns)
        title: Report title
        out_path: Output file path
        freq: 'D' for daily report (default), 'M' for monthly-aggregated report
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

    # 6) Produce report
    qs.reports.html(
        strat_excess,
        benchmark=bench_excess.to_frame("Market"),
        rf=0.0,
        periods_per_year=periods_per_year,
        output=out_path,
        title=title,
    )
    print(f"    Saved: {out_path}")
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
):
    """
    DAILY report (existing behavior). See `generate_oos_report_monthly` for monthly aggregation.
    """
    print("Generating Out-Of-Sample HTML Report (Daily)")

    # Ensure output directory exists
    ensure_dir(output_dir)

    # Extract mask where positions exist
    used_mask_oos = portfolio_result["weights"] != 0

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
    )


def generate_oos_report_monthly(
    portfolio_result,
    oos_df,
    dates_out_sample,
    output_path="outputs/oos_long_short_tearsheet_monthly.html",
    report_title=None,
    output_dir="outputs",
):
    """
    MONTHLY report (aggregates daily returns -> monthly returns by compounding).
    """
    print("Generating Out-Of-Sample HTML Report (Monthly Aggregated)")

    # Ensure output directory exists
    ensure_dir(output_dir)

    # Extract mask where positions exist
    used_mask_oos = portfolio_result["weights"] != 0

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
    )
