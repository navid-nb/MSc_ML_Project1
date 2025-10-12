import random


def split_train_and_test(df, split_pct, random_state):
    """
    Split the dataframe into in-sample (training/validation) and out-of-sample (test) sets.

    Args:
        df: DataFrame with MultiIndex (permno, date)
        split_pct: Percentage of data to use for in-sample (default: 0.80)
        random_state: Random seed for reproducibility

    Returns:
        tuple: (ins_dates, dates_out_sample, split_date)
    """
    random.seed(random_state)

    # get all unique dates
    dates_all = df.index.get_level_values("date").unique().sort_values()
    trading_days_count = len(dates_all)
    n_rows = df.shape[0]

    print("Total Data:")
    print(f"  Dates: {trading_days_count} trading days")
    print(f"  Period: {dates_all.min().date()} to {dates_all.max().date()}")
    print(f"  Rows: {n_rows:,} (stocks × dates)")

    # split: in-sample (split_pct%) vs out-of-sample (1-split_pct%)
    split_idx = int(trading_days_count * split_pct)

    # in-sample: used for feature selection, hyperparameter tuning, model development
    ins_dates = dates_all[:split_idx]

    # out-of-sample: never touched until final evaluation
    dates_out_sample = dates_all[split_idx:]

    # split date (boundary between in-sample and out-of-sample)
    split_date = dates_out_sample[0]

    print("\nData Split:")
    print("   In-Sample (Development Set):")
    print(f"   Period: {ins_dates.min().date()} to {ins_dates.max().date()}")
    print(f"   Dates: {len(ins_dates)} days ({len(ins_dates) / trading_days_count * 100:.1f}%)")
    print("   Purpose: feature selection, hyperparameter tuning, rolling CV")

    print("\n   Out-Of-Sample (Test Set):")
    print(f"   Period: {dates_out_sample.min().date()} to {dates_out_sample.max().date()}")
    print(
        f"   Dates: {len(dates_out_sample)} days ({len(dates_out_sample) / trading_days_count * 100:.1f}%)"
    )
    print("   Purpose: final performance evaluation only")

    print(f"\nSplit Date: {split_date.date()}")

    return ins_dates, dates_out_sample, split_date


def split_rolling_window(
    ins_dates, split_pct_rolling_train=0.6, split_pct_rolling_test=0.2, target_folds_count=10
):
    """
    Configure rolling window parameters for cross-validation.

    This function implements a 60/20/20 split (train/validation/test) within the in-sample data,
    creating multiple rolling windows for robust model evaluation.

    Args:
        ins_dates: DatetimeIndex of in-sample dates
        split_pct_rolling_train: Percentage of in-sample data for training window (default: 0.6)
        split_pct_rolling_test: Percentage of in-sample data for validation window (default: 0.2)
        target_folds_count: Target number of rolling windows to create (default: 10)

    Returns:
        tuple: (ins_window_size, ins_training_window_size, ins_validation_window_size, step_size, actual_folds)
    """
    # Calculate window sizes
    ins_window_size = len(ins_dates)
    ins_training_window_size = int(split_pct_rolling_train * ins_window_size)
    ins_validation_window_size = int(split_pct_rolling_test * ins_window_size)

    # Calculate step size to achieve target number of folds
    # Formula: step = (remaining_data) / (target_folds - 1)
    remaining_data = ins_window_size - ins_training_window_size - ins_validation_window_size
    step_size = max(1, remaining_data // max(1, target_folds_count - 1))

    # Calculate actual number of windows
    actual_folds = (
        ins_window_size - ins_training_window_size - ins_validation_window_size
    ) // step_size + 1

    # Print configuration
    print("Rolling Window Configuration (In-Sample Only):")
    print(
        f"   Training window: {ins_training_window_size} days (~{ins_training_window_size / 252:.1f} years, {ins_training_window_size / ins_window_size * 100:.1f}% of in-sample)"
    )
    print(
        f"   Validation window: {ins_validation_window_size} days (~{ins_validation_window_size / 252:.1f} years, {ins_validation_window_size / ins_window_size * 100:.1f}% of in-sample)"
    )
    print(f"   Step size: {step_size} days (~{step_size / 5:.1f} weeks)")
    print(f"   Target folds: {target_folds_count}")
    print(f"   Actual folds: {actual_folds}")
    print(
        f"   Total validation observations: {actual_folds * ins_validation_window_size} (across {actual_folds} overlapping folds)"
    )

    return (
        ins_window_size,
        ins_training_window_size,
        ins_validation_window_size,
        step_size,
        actual_folds,
    )
