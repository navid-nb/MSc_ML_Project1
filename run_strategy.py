import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from functions.helpers._extract import ensure_dir
from functions.helpers.allocation_strategies import apply_allocation_strategy
from functions.helpers.data_extraction import wrds_extract_raw
from functions.helpers.output_generation import make_qs_report_from_equity
from functions.helpers.portfolio_backtest import (
    calculate_performance_metrics,
    calculate_portfolio_returns,
)
from functions.helpers.split_window import split_rolling_window, split_train_and_test
from run_data import build_model_matrix_from_raw_data

# =============================================================
# 1. Data build & cleaning (CRSP/DSF/IBES/FF)
# =============================================================

raw_data = wrds_extract_raw(
    wrds_user="your-wrds-username",
    start="2010-01-01",
    end="2025-01-01",
    chunk_size=500_000,
    use_run="last",  # "new", "last", or a specific folder name (e.g. "run_20250914_133747"),
    base_dir="data",
    artifacts=[
        ("functions/migrations/001_base_extract.sql", "dsf.parquet"),
        ("functions/migrations/002_ff_factors.sql", "ff.parquet"),
        ("functions/migrations/003_ibes_statsumu.sql", "ibes_stats.parquet"),
        ("functions/migrations/004_ibes_actu.sql", "ibes_act.parquet"),
    ],
)

df = build_model_matrix_from_raw_data(
    raw_data=raw_data,
    tickers=[
        "AAPL", "NVDA", "MSFT", "AMZN", "TSLA", "GOOGL", "LLY", "WMT", "JPM", "BRK-B",
        # "V", "MA", "XOM", "ORCL", "UNH", "COST", "PG", "HD", "NFLX",
        # "JNJ", "BAC", "CRM", "QQQ", "ABBV", "KO", "CVX", "TMUS", "MRK", "CSCO",
        # "WFC", "ACN", "NOW", "TSM", "AXP", "PEP", "MCD", "IBM", "MS", "DIS",
        # "TMO", "ABT", "AMD", "ADBE", "PM", "ISRG", "GE", "GS", "INTU", "CAT",
        # "TXN", "QCOM", "RY", "VZ", "DHR", "BKNG", "T", "BLK", "SPGI",
        # "RTX", "PFE", "NEE", "HON", "CMCSA", "PGR", "AMGN", "LOW", "ANET", "UNP",
        # "SYK", "TJX", "C", "BA", "SCHW", "BSX", "KKR", "ETN",
        # "COP", "BX", "PANW", "ADP"
    ],
)

# =============================================================
# 2. Train/test split & rolling CV split
# =============================================================

# Execute the split
random_state = 42
split_pct = 0.70
ins_dates, dates_out_sample, split_date = split_train_and_test(df, split_pct, random_state)

# Rolling window size configuration for in-sample (60/20/20 Split)
# When naming variables, ins short for in-sample, oos short for out-of-sample

# Configure rolling windows
split_pct_rolling_train = 0.6  # 60% for training
split_pct_rolling_test = 0.2  # 20% for validation
target_folds_count = 10

ins_window_size, ins_training_window_size, ins_validation_window_size, step_size, actual_folds = (
    split_rolling_window(
        ins_dates,
        split_pct_rolling_train=split_pct_rolling_train,
        split_pct_rolling_test=split_pct_rolling_test,
        target_folds_count=target_folds_count,
    )
)

# =============================================================
# 3. Logistic Regression (Direction)
# =============================================================

# 3.1 Configuration
# Target Column: adj_prc_logret_lead1 = next-day log return(t -> t+1)
# We predict: will the stock go up (1) or down (0) tomorrow?

DIR_binary = (df["adj_prc_logret_lead1"] > 0).astype(int)  # 1 = up, 0 = down

# Check class balance (market neutrality ~ 50/50)
print("\nBinary Target Distribution")
print(
    f" Up (1):   {(DIR_binary == 1).sum():,} observations ({(DIR_binary == 1).mean() * 100:.1f}%)"  # noqa
)  # noqa
print(
    f" Down (0): {(DIR_binary == 0).sum():,} observations ({(DIR_binary == 0).mean() * 100:.1f}%)"  # noqa
)  # noqa
print(f" Total:    {len(DIR_binary):,} observations")

# Feature columns: everything except ticker, target, and the index columns, permno and date.
num_pred_cols = [c for c in df.columns if c not in (["ticker", "adj_prc_logret_lead1"])]
print(f"\nUsing {len(num_pred_cols)} features for prediction")

# =============================================================
# 3.2 Hyperparameter Tuning - L1 Ratio Grid
# =============================================================

# Control variable for hyperparameter tuning
HYPERPARAMETER_TUNING = True  # Set to False to use hardcoded l1_ratio=0.7

# Prepare data for logistic regression (use only in-sample data)
df_ins = df[df.index.get_level_values("date").isin(ins_dates)]
X_log_ins = df_ins[num_pred_cols]
y_log_ins = DIR_binary[df.index.get_level_values("date").isin(ins_dates)]

# Define l1_ratio grid (l1_ratio bounded [0, 1])
if HYPERPARAMETER_TUNING:
    # Run hyperparameter tuning via cross-validation
    l1_ratios = [0.7, 0.8, 0.9]  # 4 values from 0 to 1 inclusive
    C = 1
    print(f"Testing {len(l1_ratios)} l1_ratio values")
    print(f"L1 ratio range: [{min(l1_ratios):.3f}, {max(l1_ratios):.3f}]")

    # Store classification error rates for each l1_ratio
    error_rates = []

    # Cross-validation loop
    for idx, l1_ratio in enumerate(l1_ratios):
        # Progress tracking for l1_ratio
        print(f"Processing l1_ratio {idx + 1}/{len(l1_ratios)}: {l1_ratio:.3f}")
        fold_errors = []

        # Iterate through rolling windows
        for fold_idx in range(actual_folds):
            # Progress tracking for folds
            if fold_idx == 0 or (fold_idx + 1) % 2 == 0:  # Show progress every 2 folds
                print(f"  Fold {fold_idx + 1}/{actual_folds}", end="... ")

            # Calculate window indices using the pre-configured parameters
            start_idx = fold_idx * step_size
            train_end_idx = start_idx + ins_training_window_size
            val_end_idx = train_end_idx + ins_validation_window_size

            # Get date ranges from the in-sample dates
            train_dates = ins_dates[start_idx:train_end_idx]
            val_dates = ins_dates[train_end_idx:val_end_idx]

            # Split data using date filters
            train_idx = df_ins.index.get_level_values("date").isin(train_dates)
            val_idx = df_ins.index.get_level_values("date").isin(val_dates)

            X_train, y_train = X_log_ins[train_idx], y_log_ins[train_idx]
            X_val, y_val = X_log_ins[val_idx], y_log_ins[val_idx]

            # Create pipeline with scaling and logistic regression
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            penalty="elasticnet",
                            C=C,
                            l1_ratio=l1_ratio,
                            solver="saga",
                            max_iter=5000,
                            tol=1e-4,
                            random_state=42,
                        ),
                    ),
                ]
            )

            # Fit and predict
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)

            # Calculate error rate (1 - accuracy)
            accuracy = (y_pred == y_val).mean()  # noqa
            error_rate = 1 - accuracy
            fold_errors.append(error_rate)

        # Average error rate across folds
        avg_error = np.mean(fold_errors)
        error_rates.append(avg_error)
        print(f"Completed l1_ratio {idx + 1}/{len(l1_ratios)}")

    print(f"\nCompleted validation for all {len(l1_ratios)} l1_ratio values")

    # Find optimal l1_ratio (minimum error rate)
    optimal_idx = np.argmin(error_rates)
    l1_ratio_star = l1_ratios[optimal_idx]
    min_error = error_rates[optimal_idx]

    print("\nLogistic Regression - Optimal Hyperparameters:")
    print(f"  l1_ratio* = {l1_ratio_star:.3f}")
    print(f"  Minimum Average Classification Error Rate = {min_error:.4f}")
    print(f"  Validation Accuracy = {1 - min_error:.4f}")

    # Plot error rates vs l1_ratio
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(l1_ratios, error_rates, marker="o", markersize=4)
    plt.axvline(l1_ratio_star, color="r", linestyle="--", label=f"l1_ratio* = {l1_ratio_star:.3f}")
    plt.xlabel("L1 Ratio")
    plt.ylabel("Average Classification Error Rate")
    plt.title("Logistic Regression: Classification Error vs L1 Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

else:
    # Use hardcoded l1_ratio value
    l1_ratio_star = 0.7
    C = 0.1
    print(f"Skipping hyperparameter tuning. Using hardcoded l1_ratio = {l1_ratio_star}")

# =============================================================
# 3.3 Final Model Estimation and Out-of-Sample Evaluation
# =============================================================

# Prepare full in-sample (training) and out-of-sample (test) data
df_oos = df[df.index.get_level_values("date").isin(dates_out_sample)]

X_train_full = X_log_ins
y_train_full = y_log_ins
X_test = df_oos[num_pred_cols]
y_test = DIR_binary[df.index.get_level_values("date").isin(dates_out_sample)]

# Fit final model on entire training dataset using optimal l1_ratio
final_pipeline_log = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                penalty="elasticnet",
                C=C,
                l1_ratio=l1_ratio_star,  # Use optimal l1_ratio
                solver="saga",
                max_iter=5000,
                tol=1e-4,  # More lenient tolerance
                random_state=42,
            ),
        ),
    ]
)

final_pipeline_log.fit(X_train_full, y_train_full)

# Get coefficients
coefficients_log = final_pipeline_log.named_steps["logistic"].coef_[0]
intercept_log = final_pipeline_log.named_steps["logistic"].intercept_[0]

print(f"Final model fitted on {len(X_train_full):,} training observations")
print(f"Test set contains {len(X_test):,} observations")
print(f"\nIntercept: {intercept_log:.6f}")
print(
    f"Number of non-zero coefficients: {(coefficients_log != 0).sum()}/{len(coefficients_log)}"  # noqa
)  # noqa

# Generate predictions on test set
y_pred_test = final_pipeline_log.predict(X_test)
y_pred_proba_test = final_pipeline_log.predict_proba(X_test)[:, 1]

# Calculate performance metrics
test_accuracy = (y_pred_test == y_test).mean()  # noqa
test_error = 1 - test_accuracy

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)

print("=" * 60)
print("LOGISTIC REGRESSION - OUT-OF-SAMPLE PERFORMANCE")
print("=" * 60)
print(f"\nTest Set Accuracy:           {test_accuracy:.4f}")
print(f"Test Set Error Rate:         {test_error:.4f}")
print("\nConfusion Matrix:")
print("                 Predicted Down  Predicted Up")
print(f"Actual Down      {conf_matrix[0, 0]:>14,}  {conf_matrix[0, 1]:>12,}")
print(f"Actual Up        {conf_matrix[1, 0]:>14,}  {conf_matrix[1, 1]:>12,}")

print(f"\n{classification_report(y_test, y_pred_test, target_names=['Down (0)', 'Up (1)'])}")

# Analyze most influential predictors
coef_df = pd.DataFrame({"feature": num_pred_cols, "coefficient": coefficients_log})
coef_df = coef_df[coef_df["coefficient"] != 0].copy()
coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

print("\nTop 10 Most Influential Predictors (Non-Zero Coefficients):")
print(coef_df.head(10).to_string(index=False))


# =============================================================
# 4. Linear Regression (Magnitude)
# =============================================================

# 4.1 Configuration
# Target Column: adj_prc_logret_lead1 = next-day log return(t -> t+1)
# We predict the MAGNITUDE of tomorrow's return (continuous value)

# Continuous target (actual log returns, not binarized)
y_continuous = df["adj_prc_logret_lead1"]

# Distribution statistics
print("\nContinuous Target Distribution")
print(f"  Mean:   {y_continuous.mean():>10.6f}")
print(f"  Std:    {y_continuous.std():>10.6f}")
print(f"  Range:  [{y_continuous.min():.6f}, {y_continuous.max():.6f}]")
print(f"  Total:  {len(y_continuous):,} observations")
print(f"\nUsing {len(num_pred_cols)} features for prediction (same as logistic regression)")

# =============================================================
# 4.2 Hyperparameter Tuning - L1 Ratio Grid
# =============================================================

# Control variable for hyperparameter tuning
HYPERPARAMETER_TUNING_LINEAR = True  # Set to False to use hardcoded l1_ratio=0.7

# Prepare data for linear regression (use only in-sample data)
df_ins = df[df.index.get_level_values("date").isin(ins_dates)]
X_lin_ins = df_ins[num_pred_cols]
y_lin_ins = df_ins["adj_prc_logret_lead1"]

# Define l1_ratio grid (l1_ratio bounded [0, 1])
if HYPERPARAMETER_TUNING_LINEAR:
    # Run hyperparameter tuning via cross-validation
    l1_ratios_lin = [0.5, 0.6, 0.7]  # 4 values from 0 to 1 inclusive
    alpha_fixed = 0.001  # Fixed regularization strength
    print(f"Testing {len(l1_ratios_lin)} l1_ratio values")
    print(f"L1 ratio range: [{min(l1_ratios_lin):.3f}, {max(l1_ratios_lin):.3f}]")
    print(f"Alpha (fixed): {alpha_fixed}")

    # Store RMSE for each l1_ratio
    rmse_values = []

    # Cross-validation loop
    for idx, l1_ratio in enumerate(l1_ratios_lin):
        # Progress tracking for l1_ratio
        print(f"Processing l1_ratio {idx + 1}/{len(l1_ratios_lin)}: {l1_ratio:.3f}")
        fold_rmse = []

        # Iterate through rolling windows
        for fold_idx in range(actual_folds):
            # Progress tracking for folds
            if fold_idx == 0 or (fold_idx + 1) % 2 == 0:  # Show progress every 2 folds
                print(f"  Fold {fold_idx + 1}/{actual_folds}", end="... ")

            # Calculate window indices using the pre-configured parameters
            start_idx = fold_idx * step_size
            train_end_idx = start_idx + ins_training_window_size
            val_end_idx = train_end_idx + ins_validation_window_size

            # Get date ranges from the in-sample dates
            train_dates = ins_dates[start_idx:train_end_idx]
            val_dates = ins_dates[train_end_idx:val_end_idx]

            # Split data using date filters
            train_idx = df_ins.index.get_level_values("date").isin(train_dates)
            val_idx = df_ins.index.get_level_values("date").isin(val_dates)

            X_train, y_train = X_lin_ins[train_idx], y_lin_ins[train_idx]
            X_val, y_val = X_lin_ins[val_idx], y_lin_ins[val_idx]

            # Create pipeline with scaling and ElasticNet
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lasso",
                        ElasticNet(
                            alpha=alpha_fixed,  # Fixed regularization strength
                            l1_ratio=l1_ratio,  # Tune l1_ratio
                            max_iter=10000,
                            tol=1e-4,
                            random_state=42,
                        ),
                    ),
                ]
            )

            # Fit and predict
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            fold_rmse.append(rmse)

        # Average RMSE across folds
        avg_rmse = np.mean(fold_rmse)
        rmse_values.append(avg_rmse)
        print(f"Completed l1_ratio {idx + 1}/{len(l1_ratios_lin)}")

    print(f"\nCompleted validation for all {len(l1_ratios_lin)} l1_ratio values")

    # Find optimal l1_ratio (minimum RMSE)
    optimal_idx = np.argmin(rmse_values)
    l1_ratio_star_lin = l1_ratios_lin[optimal_idx]
    min_rmse = rmse_values[optimal_idx]

    print("\nLinear Regression - Optimal Hyperparameters:")
    print(f"  l1_ratio* = {l1_ratio_star_lin:.3f}")
    print(f"  Minimum Average RMSE = {min_rmse:.6f}")

    # Plot RMSE vs l1_ratio
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(l1_ratios_lin, rmse_values, marker="o", markersize=4)
    plt.axvline(
        l1_ratio_star_lin, color="r", linestyle="--", label=f"l1_ratio* = {l1_ratio_star_lin:.3f}"
    )
    plt.xlabel("L1 Ratio")
    plt.ylabel("Average RMSE")
    plt.title("Linear Regression: RMSE vs L1 Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

else:
    # Use hardcoded l1_ratio value
    l1_ratio_star_lin = 0.7
    alpha_fixed = 0.001
    print(f"Skipping hyperparameter tuning. Using hardcoded l1_ratio = {l1_ratio_star_lin}")

# =============================================================
# 4.3 Final Model Estimation and Out-of-Sample Evaluation
# =============================================================

# Prepare full in-sample (training) and out-of-sample (test) data
df_oos = df[df.index.get_level_values("date").isin(dates_out_sample)]

X_train_full_lin = X_lin_ins
y_train_full_lin = y_lin_ins
X_test_lin = df_oos[num_pred_cols]
y_test_lin = df_oos["adj_prc_logret_lead1"]

# Fit final model on entire training dataset using optimal l1_ratio
final_pipeline_lin = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "lasso",
            ElasticNet(
                alpha=alpha_fixed,  # Fixed regularization strength
                l1_ratio=l1_ratio_star_lin,  # Use optimal l1_ratio
                max_iter=10000,
                tol=1e-4,
                random_state=42,
            ),
        ),
    ]
)

final_pipeline_lin.fit(X_train_full_lin, y_train_full_lin)

# Get coefficients
coefficients_lin = final_pipeline_lin.named_steps["lasso"].coef_
intercept_lin = final_pipeline_lin.named_steps["lasso"].intercept_

print(f"Final model fitted on {len(X_train_full_lin):,} training observations")
print(f"Test set contains {len(X_test_lin):,} observations")
print(f"\nIntercept: {intercept_lin:.6f}")
print(
    f"Number of non-zero coefficients: {(coefficients_lin != 0).sum()}/{len(coefficients_lin)}"  # noqa
)  # noqa

# Generate predictions on test set
y_pred_test_lin = final_pipeline_lin.predict(X_test_lin)

# Calculate performance metrics
test_rmse = np.sqrt(mean_squared_error(y_test_lin, y_pred_test_lin))
test_r2 = r2_score(y_test_lin, y_pred_test_lin)
test_mae = np.mean(np.abs(y_test_lin - y_pred_test_lin))

print("=" * 60)
print("LINEAR REGRESSION - OUT-OF-SAMPLE PERFORMANCE")
print("=" * 60)
print(f"\nTest Set RMSE:               {test_rmse:.6f}")
print(f"Test Set R²:                 {test_r2:.6f}")
print(f"Test Set MAE:                {test_mae:.6f}")
print("\nBaseline (predicting mean):")
baseline_rmse = np.sqrt(mean_squared_error(y_test_lin, [y_train_full_lin.mean()] * len(y_test_lin)))
print(f"Baseline RMSE:               {baseline_rmse:.6f}")
print(f"Improvement over baseline:   {((baseline_rmse - test_rmse) / baseline_rmse * 100):.2f}%")

# Analyze most influential predictors
coef_df_lin = pd.DataFrame({"feature": num_pred_cols, "coefficient": coefficients_lin})
coef_df_lin = coef_df_lin[coef_df_lin["coefficient"] != 0].copy()
coef_df_lin["abs_coefficient"] = coef_df_lin["coefficient"].abs()
coef_df_lin = coef_df_lin.sort_values("abs_coefficient", ascending=False)

print("\nTop 10 Most Influential Predictors (Non-Zero Coefficients):")
print(coef_df_lin.head(10).to_string(index=False))

# =============================================================
# 5. Signal Confirmation & Trading Universe Selection
# =============================================================

# Configuration: Thresholds for signal generation
# Logistic: LONG if prob_up > threshold; SHORT if prob_up < (1 - threshold)
# Linear: LONG if expected_return > threshold; SHORT if expected_return < -threshold
PROB_UP_THRESHOLD_LONG = 0.55
PROB_UP_THRESHOLD_SHORT = 0.55
EXPECTED_RETURN_THRESHOLD_LONG = 0.001
EXPECTED_RETURN_THRESHOLD_SHORT = 0.001

# Generate predictions for all data (in-sample + out-of-sample)
X_full_logistic = df[num_pred_cols].copy()
prob_up_full = final_pipeline_log.predict_proba(X_full_logistic)[:, 1]
logistic_score_full = 2 * prob_up_full - 1  # Scale to [-1, 1]

X_full_linear = df[num_pred_cols].copy()
expected_return_full = final_pipeline_lin.predict(X_full_linear)

# Store predictions in DataFrame
df_signals = df.copy()
df_signals["prob_up"] = prob_up_full
df_signals["prob_down"] = 1 - prob_up_full
df_signals["logistic_score"] = logistic_score_full
df_signals["expected_return"] = expected_return_full

# Define trading signals from each model
# Logistic Regression Signals (direction based on probability)
df_signals["logistic_signal_long"] = df_signals["prob_up"] > PROB_UP_THRESHOLD_LONG
df_signals["logistic_signal_short"] = df_signals["prob_up"] < (1 - PROB_UP_THRESHOLD_SHORT)

# Linear Regression Signals (direction based on expected return)
df_signals["linear_signal_long"] = df_signals["expected_return"] > EXPECTED_RETURN_THRESHOLD_LONG
df_signals["linear_signal_short"] = df_signals["expected_return"] < -EXPECTED_RETURN_THRESHOLD_SHORT

# Ensemble Agreement: Both models must agree on direction
df_signals["agreed_long"] = df_signals["logistic_signal_long"] & df_signals["linear_signal_long"]
df_signals["agreed_short"] = df_signals["logistic_signal_short"] & df_signals["linear_signal_short"]
df_signals["agreed_any"] = df_signals["agreed_long"] | df_signals["agreed_short"]
df_signals["disagreed"] = ~df_signals["agreed_any"]

# Create ensemble score for ranking (0 if models disagree)
df_signals["ensemble_score"] = df_signals["logistic_score"].copy()
df_signals.loc[df_signals["disagreed"], "ensemble_score"] = 0.0

# ============================================================================
# Output 1: Agreement Statistics
# ============================================================================
print("=" * 80)
print("Ensemble Agreement (Both Models Must Agree)")
print("=" * 80)
print("\nAgreement Statistics:")
print(
    f"  Both Agree LONG:   {df_signals['agreed_long'].sum():>6,} ({df_signals['agreed_long'].mean()*100:>5.1f}%)"
)
print(
    f"  Both Agree SHORT:  {df_signals['agreed_short'].sum():>6,} ({df_signals['agreed_short'].mean()*100:>5.1f}%)"
)
print("  -------------------------------------")
print(
    f"  Total Agreement:   {df_signals['agreed_any'].sum():>6,} ({df_signals['agreed_any'].mean()*100:>5.1f}%)"
)
print(
    f"  Disagreement:      {df_signals['disagreed'].sum():>6,} ({df_signals['disagreed'].mean()*100:>5.1f}%)"
)

# ============================================================================
# Output 2: Trading Universe Breakdown
# ============================================================================
long_universe = df_signals[df_signals["agreed_long"]].copy()
short_universe = df_signals[df_signals["agreed_short"]].copy()

print("\n" + "=" * 80)
print("Trading Universe Breakdown by Direction")
print("=" * 80)

print(f"\nLONG Universe:  {len(long_universe):>6,} observations")
if len(long_universe) > 0:
    print(f"   Mean prob(up):       {long_universe['prob_up'].mean():.4f}")
    print(f"   Mean E[R]:           {long_universe['expected_return'].mean():.6f}")
    print(f"   Mean ensemble score: {long_universe['ensemble_score'].mean():.4f}")
else:
    print("   (No long candidates)")

print(f"\nSHORT Universe: {len(short_universe):>6,} observations")
if len(short_universe) > 0:
    print(f"   Mean prob(up):       {short_universe['prob_up'].mean():.4f}")
    print(f"   Mean E[R]:           {short_universe['expected_return'].mean():.6f}")
    print(f"   Mean ensemble score: {short_universe['ensemble_score'].mean():.4f}")
else:
    print("   (No short candidates)")


# Variables created for downstream use:
# - df_signals:       Full DataFrame with predictions, signals, and ensemble_score
# - long_universe:    Filtered DataFrame for long candidates
# - short_universe:   Filtered DataFrame for short candidates
# - excluded_universe: Filtered DataFrame for disagreements (not traded)

# =============================================================
# 6. Strategy Selection (In-Sample Performance)
# =============================================================

print("=" * 80)
print("STRATEGY SELECTION ON IN-SAMPLE DATA")
print("=" * 80)
print("\nNote: Strategy selection uses in-sample Sharpe ratio")
print("      This simulates a realistic scenario where you only have historical data")
print("      Expect performance degradation when applying to out-of-sample data")

# =============================================================
# 6.1 Define Methods & Strategies
# =============================================================

SCORING_METHODS = {
    "S1": ("score_S1", "p · μ (Product)"),
    "S2": ("score_S2", "(p - 0.5) · μ (Margin-aware)"),
    "S6": ("score_S6", "(2p-1) · |μ| (Directional)"),
}

ALLOCATION_STRATEGIES = ["A1", "A2", "A3", "A4", "A5"]

ALLOCATION_DESCRIPTIONS = {
    "A1": "Equal-Weighted",
    "A2": "Rank-Weighted",
    "A3": "Top/Bottom Quantile",
    "A4": "Score-Weighted (Threshold)",
    "A5": "Volatility-Scaled",
}

# Configuration
LONG_TARGET = 0.5
SHORT_TARGET = 0.5
MAX_POSITION_SIZE = 0.05
QUANTILE_LONG_PCT = 0.20
QUANTILE_SHORT_PCT = 0.20

print("\nConfiguration:")
print(f"  Scoring methods: {len(SCORING_METHODS)}")
print(f"  Allocation strategies: {len(ALLOCATION_STRATEGIES)}")
print(f"  Total combinations: {len(SCORING_METHODS) * len(ALLOCATION_STRATEGIES)}")

# =============================================================
# 6.2 Test All Combinations (In-Sample)
# =============================================================

print(
    f"\nTesting {len(SCORING_METHODS)} × {len(ALLOCATION_STRATEGIES)} = {len(SCORING_METHODS) * len(ALLOCATION_STRATEGIES)} combinations on in-sample data..."
)

# Filter to in-sample data
df_ins_signals = df_signals[df_signals.index.get_level_values("date").isin(ins_dates)].copy()

# =============================================================
# Calculate Scoring Methods (S1, S2, S6)
# =============================================================

# Extract model outputs
p_ins = df_ins_signals["prob_up"]
mu_ins = df_ins_signals["expected_return"]

# Calculate scores
df_ins_signals["score_S1"] = p_ins * mu_ins  # Product
df_ins_signals["score_S2"] = (p_ins - 0.5) * mu_ins  # Margin-aware
df_ins_signals["score_S6"] = (2 * p_ins - 1) * mu_ins.abs()  # Directional

# Use pre-calculated annualized volatility (required for A5 - Inverse-Vol strategy)
# This was already computed in feature engineering as 20-day rolling std × sqrt(252)
if "anualized_volatility_20d" in df_ins_signals.columns:
    df_ins_signals["volatility"] = df_ins_signals["anualized_volatility_20d"] / np.sqrt(
        252
    )  # Convert back to daily
else:
    # Fallback: calculate if not present
    df_ins_signals["volatility"] = df_ins_signals.groupby(level="permno")[
        "adj_prc_logret"
    ].transform(lambda x: x.rolling(window=20, min_periods=5).std())
df_ins_signals["volatility"] = df_ins_signals.groupby(level="date")["volatility"].transform(
    lambda x: x.fillna(x.median())
)

# Use 20-day rolling std of returns, fill missing with cross-sectional median
df_ins_signals["volatility"] = df_ins_signals.groupby(level="permno")["adj_prc_logret"].transform(
    lambda x: x.rolling(window=20, min_periods=5).std()
)
df_ins_signals["volatility"] = df_ins_signals.groupby(level="date")["volatility"].transform(
    lambda x: x.fillna(x.median())
)

print(
    f"  In-sample period: {df_ins_signals.index.get_level_values('date').min()} to {df_ins_signals.index.get_level_values('date').max()}"
)
print(f"  Observations: {len(df_ins_signals):,}")

# Import functions

# Storage
results = []
errors = []

# Test each combination
combo_count = 0
for score_name, (score_col, score_desc) in SCORING_METHODS.items():
    for alloc_strategy in ALLOCATION_STRATEGIES:
        combo_count += 1

        try:
            scores = df_ins_signals[score_col].copy()
            long_mask = df_ins_signals["agreed_long"]
            short_mask = df_ins_signals["agreed_short"]

            df_ins_work = df_ins_signals.copy()
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
                    "scoring_desc": score_desc,
                    "allocation_strategy": alloc_strategy,
                    "allocation_desc": ALLOCATION_DESCRIPTIONS[alloc_strategy],
                    "n_trades": (weights_series != 0).sum(),  # noqa
                    "avg_n_positions": (weights_series != 0)
                    .groupby(level="date")  # noqa
                    .sum()
                    .mean(),  # noqa
                    "total_return": metrics["total_return"],
                    "ann_return": metrics["ann_return"],
                    "ann_vol": metrics["ann_vol"],
                    "sharpe": metrics["sharpe"],
                    "sortino": metrics["sortino"],
                    "max_drawdown": metrics["max_drawdown"],
                    "calmar": metrics["calmar"],
                    "win_rate": metrics["win_rate"],
                }
            )

            print(
                f"  [{combo_count:2d}/{len(SCORING_METHODS)*len(ALLOCATION_STRATEGIES):2d}] {score_name} + {alloc_strategy}: Sharpe = {metrics['sharpe']:.3f}"
            )

        except Exception as e:
            print(
                f"  [{combo_count:2d}/{len(SCORING_METHODS)*len(ALLOCATION_STRATEGIES):2d}] {score_name} + {alloc_strategy}: ERROR - {str(e)[:50]}"
            )
            errors.append({"scoring": score_name, "allocation": alloc_strategy, "error": str(e)})
            continue

# =============================================================
# 6.3 Display Results
# =============================================================

print("=" * 80)
print("IN-SAMPLE RESULTS")
print("=" * 80)

results_df = pd.DataFrame(results)

if len(results_df) > 0:
    print(
        f"\nSuccessfully tested: {len(results_df)}/{len(SCORING_METHODS) * len(ALLOCATION_STRATEGIES)} combinations"
    )

    # Top 10 strategies
    print("\nTop 10 Strategies by Sharpe Ratio:")
    print("-" * 80)
    top_strategies = results_df.nlargest(10, "sharpe")[
        ["scoring_method", "allocation_strategy", "sharpe", "ann_return", "ann_vol"]
    ]
    print(top_strategies.to_string(index=False))

    # Comparison matrix
    print("\nSharpe Ratio Matrix:")
    print("-" * 80)
    pivot_sharpe = results_df.pivot(
        index="allocation_strategy", columns="scoring_method", values="sharpe"
    )
    print(pivot_sharpe.to_string(float_format=lambda x: f"{x:.3f}"))

    # Select optimal
    print("=" * 80)
    print("OPTIMAL STRATEGY (Selected on In-Sample Data)")
    print("=" * 80)

    best_combo = results_df.loc[results_df["sharpe"].idxmax()]
    print(f"\n  Scoring Method:      {best_combo['scoring_method']} - {best_combo['scoring_desc']}")
    print(
        f"  Allocation Strategy: {best_combo['allocation_strategy']} - {best_combo['allocation_desc']}"
    )
    print("\n  In-Sample Performance:")
    print(f"    Sharpe Ratio:        {best_combo['sharpe']:.3f}")
    print(f"    Ann. Return:         {best_combo['ann_return']:.2f}%")
    print(f"    Ann. Volatility:     {best_combo['ann_vol']:.2f}%")
    print(f"    Max Drawdown:        {best_combo['max_drawdown']:.2f}%")

    # Store for Section 7
    strategy_comparison_results = results_df
    optimal_scoring_method = best_combo["scoring_method"]
    optimal_allocation_strategy = best_combo["allocation_strategy"]
    optimal_score_col = SCORING_METHODS[optimal_scoring_method][0]

    # Overfitting note
    print("=" * 80)
    print("OVERFITTING EXPECTATIONS")
    print("=" * 80)
    print(f"  Tested {len(results_df)} strategy combinations")
    print("  Expected out-of-sample degradation: 15-30%")
    print(
        f"  If IS Sharpe = {best_combo['sharpe']:.3f}, expect OOS Sharpe between {best_combo['sharpe']*0.70:.3f} and {best_combo['sharpe']*0.85:.3f}"
    )
    print("\n  Section 7 will evaluate this strategy on out-of-sample data")

else:
    print("\nNo successful strategy combinations")

if len(errors) > 0:
    print(f"\n{len(errors)} combinations failed")


# =============================================================
# 7. Out-of-Sample Evaluation (Final Performance Test)
# =============================================================

# CONTROL PARAMETER: Choose evaluation and output scope
# This controls BOTH Section 7 (evaluation) AND Section 9 (report generation)
# Options: "optimal" (best strategy only), "top5" (top 5 strategies), "all" (all 15 combinations)
EVALUATION_SCOPE = "top5"  # Change to "top5" or "all" to evaluate/output multiple strategies

print("=" * 80)
print("OUT-OF-SAMPLE EVALUATION")
print("=" * 80)

# =============================================================
# 7.1 Prepare Out-of-Sample Data
# =============================================================

# Filter to out-of-sample data
df_oos_signals = df_signals[df_signals.index.get_level_values("date").isin(dates_out_sample)].copy()

# Calculate scoring methods on OOS data
p_oos = df_oos_signals["prob_up"]
mu_oos = df_oos_signals["expected_return"]

df_oos_signals["score_S1"] = p_oos * mu_oos
df_oos_signals["score_S2"] = (p_oos - 0.5) * mu_oos
df_oos_signals["score_S6"] = (2 * p_oos - 1) * mu_oos.abs()

# Use pre-calculated annualized volatility
if "anualized_volatility_20d" in df_oos_signals.columns:
    df_oos_signals["volatility"] = df_oos_signals["anualized_volatility_20d"] / np.sqrt(252)
else:
    df_oos_signals["volatility"] = df_oos_signals.groupby(level="permno")[
        "adj_prc_logret"
    ].transform(lambda x: x.rolling(window=20, min_periods=5).std())
df_oos_signals["volatility"] = df_oos_signals.groupby(level="date")["volatility"].transform(
    lambda x: x.fillna(x.median())
)

print(
    f"\nOut-of-sample period: {df_oos_signals.index.get_level_values('date').min()} to {df_oos_signals.index.get_level_values('date').max()}"
)
print(f"Observations: {len(df_oos_signals):,}")

# =============================================================
# 7.2 Determine Which Strategies to Evaluate
# =============================================================

if EVALUATION_SCOPE == "optimal":
    strategies_to_evaluate = [(optimal_scoring_method, optimal_allocation_strategy)]
    print("\nScope: OPTIMAL strategy only")
    print(f"  Strategy: {optimal_scoring_method} + {optimal_allocation_strategy}")
elif EVALUATION_SCOPE == "top5":
    # Get top 5 from in-sample results
    top5 = strategy_comparison_results.nlargest(5, "sharpe")[
        ["scoring_method", "allocation_strategy"]
    ]
    strategies_to_evaluate = list(top5.itertuples(index=False, name=None))
    print("\nScope: TOP 5 strategies")
    for i, (sm, as_) in enumerate(strategies_to_evaluate, 1):
        print(f"  {i}. {sm} + {as_}")
elif EVALUATION_SCOPE == "all":
    # Evaluate all combinations
    strategies_to_evaluate = []
    for sm in SCORING_METHODS.keys():
        for as_ in ALLOCATION_STRATEGIES:
            strategies_to_evaluate.append((sm, as_))
    print(f"\nScope: ALL {len(strategies_to_evaluate)} combinations")
else:
    raise ValueError(f"Invalid EVALUATION_SCOPE: {EVALUATION_SCOPE}")

# =============================================================
# 7.3 Evaluate Each Strategy on Out-of-Sample Data
# =============================================================

print("=" * 80)
print("EVALUATING STRATEGIES OUT-OF-SAMPLE")
print("=" * 80)

oos_results = []

for idx_strat, (scoring_method, allocation_strategy) in enumerate(strategies_to_evaluate, 1):
    try:
        # Get score column
        score_col = SCORING_METHODS[scoring_method][0]
        scores_oos = df_oos_signals[score_col].copy()
        long_mask_oos = df_oos_signals["agreed_long"]
        short_mask_oos = df_oos_signals["agreed_short"]

        # Calculate weights for each date
        df_oos_work = df_oos_signals.copy()
        weights_all_oos = []

        unique_dates = df_oos_work.index.get_level_values("date").unique()

        for date in unique_dates:
            date_mask = df_oos_work.index.get_level_values("date") == date

            # Allocation parameters
            alloc_params = {
                "long_target": LONG_TARGET,
                "short_target": SHORT_TARGET,
                "max_position_size": MAX_POSITION_SIZE,
            }

            if allocation_strategy == "A3":
                alloc_params["quantile_long_pct"] = QUANTILE_LONG_PCT
                alloc_params["quantile_short_pct"] = QUANTILE_SHORT_PCT

            kwargs = {}
            if allocation_strategy == "A5":
                kwargs["volatility"] = df_oos_work.loc[date_mask, "volatility"]

            weights_date = apply_allocation_strategy(
                strategy_name=allocation_strategy,
                scores=scores_oos[date_mask],
                long_mask=long_mask_oos[date_mask],
                short_mask=short_mask_oos[date_mask],
                **alloc_params,
                **kwargs,
            )

            weights_all_oos.append(weights_date)

        # Combine weights
        weights_series_oos = pd.concat(weights_all_oos)

        # Add weights to dataframe
        df_oos_work["portfolio_weights"] = weights_series_oos

        # Calculate returns
        portfolio_returns_oos = calculate_portfolio_returns(
            df=df_oos_work,
            weights_col="portfolio_weights",
            returns_col="adj_prc_logret_lead1",
            date_col="date",
        )

        # Calculate metrics
        metrics_oos = calculate_performance_metrics(
            returns=portfolio_returns_oos, rf_rate=0.0, periods_per_year=252
        )

        # Store results
        oos_results.append(
            {
                "scoring_method": scoring_method,
                "allocation_strategy": allocation_strategy,
                "sharpe_oos": metrics_oos["sharpe"],
                "ann_return_oos": metrics_oos["ann_return"],
                "ann_vol_oos": metrics_oos["ann_vol"],
                "max_dd_oos": metrics_oos["max_drawdown"],
                "portfolio_returns": portfolio_returns_oos,
            }
        )

        print(
            f"  [{idx_strat:2}/{len(strategies_to_evaluate)}] {scoring_method} + {allocation_strategy}: Sharpe = {metrics_oos['sharpe']:.3f}"
        )

    except Exception as e:
        print(
            f"  [{idx_strat:2}/{len(strategies_to_evaluate)}] {scoring_method} + {allocation_strategy}: ERROR - {str(e)}"
        )

# Create results DataFrame
oos_results_df = pd.DataFrame(oos_results)

print("=" * 80)
print("OUT-OF-SAMPLE RESULTS")
print("=" * 80)
print(f"\nSuccessfully evaluated: {len(oos_results_df)}/{len(strategies_to_evaluate)} strategies")

if len(oos_results_df) > 0:
    print("\nTop strategies by OOS Sharpe Ratio:")
    print("-" * 80)
    top_display = oos_results_df.nlargest(min(10, len(oos_results_df)), "sharpe_oos")
    print(
        top_display[
            ["scoring_method", "allocation_strategy", "sharpe_oos", "ann_return_oos", "ann_vol_oos"]
        ].to_string(index=False)
    )

    # Identify best OOS strategy
    best_oos_idx = oos_results_df["sharpe_oos"].idxmax()
    best_oos = oos_results_df.loc[best_oos_idx]

    print("=" * 80)
    print("BEST OUT-OF-SAMPLE STRATEGY")
    print("=" * 80)
    print(f"  Scoring Method:      {best_oos['scoring_method']}")
    print(f"  Allocation Strategy: {best_oos['allocation_strategy']}")
    print(f"  OOS Sharpe Ratio:    {best_oos['sharpe_oos']:.3f}")
    print(f"  OOS Ann. Return:     {best_oos['ann_return_oos']:.2f}%")
    print(f"  OOS Ann. Volatility: {best_oos['ann_vol_oos']:.2f}%")
    print(f"  OOS Max Drawdown:    {best_oos['max_dd_oos']:.2f}%")

    # Compare to in-sample selection
    print("=" * 80)
    print("IN-SAMPLE VS OUT-OF-SAMPLE COMPARISON")
    print("=" * 80)
    print(f"\nIn-Sample Selection: {optimal_scoring_method} + {optimal_allocation_strategy}")

    # Get IS performance for optimal strategy
    optimal_is = strategy_comparison_results[
        (strategy_comparison_results["scoring_method"] == optimal_scoring_method)
        & (strategy_comparison_results["allocation_strategy"] == optimal_allocation_strategy)
    ].iloc[0]

    # Get OOS performance for optimal strategy
    optimal_oos = (
        oos_results_df[
            (oos_results_df["scoring_method"] == optimal_scoring_method)
            & (oos_results_df["allocation_strategy"] == optimal_allocation_strategy)
        ].iloc[0]
        if len(
            oos_results_df[
                (oos_results_df["scoring_method"] == optimal_scoring_method)
                & (oos_results_df["allocation_strategy"] == optimal_allocation_strategy)
            ]
        )
        > 0
        else None
    )

    if optimal_oos is not None:
        print(f"\n{'Metric':<25} {'In-Sample':>15} {'Out-of-Sample':>15} {'Degradation':>12}")
        print("-" * 80)

        sharpe_change = (
            (optimal_oos["sharpe_oos"] - optimal_is["sharpe"]) / optimal_is["sharpe"] * 100
        )
        return_change = (
            (optimal_oos["ann_return_oos"] - optimal_is["ann_return"])
            / abs(optimal_is["ann_return"])
            * 100
        )
        vol_change = (
            (optimal_oos["ann_vol_oos"] - optimal_is["ann_vol"]) / optimal_is["ann_vol"] * 100
        )

        print(
            f"{'Sharpe Ratio':<25} {optimal_is['sharpe']:>15.3f} {optimal_oos['sharpe_oos']:>15.3f} {sharpe_change:>11.1f}%"
        )
        print(
            f"{'Ann. Return (%)':<25} {optimal_is['ann_return']:>15.2f} {optimal_oos['ann_return_oos']:>15.2f} {return_change:>11.1f}%"
        )
        print(
            f"{'Ann. Volatility (%)':<25} {optimal_is['ann_vol']:>15.2f} {optimal_oos['ann_vol_oos']:>15.2f} {vol_change:>11.1f}%"
        )

        print("\nAssessment:")
        sharpe_deg = abs(sharpe_change)
        if sharpe_deg < 15:
            assessment = "EXCELLENT - Minimal degradation"
        elif sharpe_deg < 30:
            assessment = "GOOD - Expected range"
        elif sharpe_deg < 50:
            assessment = "ACCEPTABLE - Some overfitting"
        else:
            assessment = "CONCERNING - Severe degradation"
        print(f"  {assessment} (Sharpe degradation: {sharpe_deg:.1f}%)")

print("=" * 80)

# =============================================================
# 9. Generate QuantStats HTML Reports\n
# =============================================================

# Note: Uses EVALUATION_SCOPE from Section 7 (must run Section 7 first)\n

print("=" * 80)
print("GENERATING QUANTSTATS HTML REPORTS")
print("=" * 80)

# Map allocation strategy codes to names
ALLOCATION_STRATEGY_NAMES = {
    "A1": "EqualWeight",
    "A2": "RankWeighted",
    "A3": "Quantile",
    "A4": "ScoreWeighted",
    "A5": "InverseVol",
}

# Determine which strategies to output
if EVALUATION_SCOPE == "optimal":
    if "oos_results_df" in locals():
        strategies_to_output = [oos_results_df.loc[oos_results_df["sharpe_oos"].idxmax()]]
    else:
        strategies_to_output = [
            {
                "scoring_method": optimal_scoring_method,
                "allocation_strategy": optimal_allocation_strategy,
                "portfolio_returns": portfolio_returns_oos,
            }
        ]
    print("\nScope: OPTIMAL strategy (1 report)")
elif EVALUATION_SCOPE == "top5":
    if "oos_results_df" not in locals():
        raise ValueError("Section 7 must be run with EVALUATION_SCOPE='top5' or 'all' first")
    strategies_to_output = oos_results_df.nlargest(5, "sharpe_oos").to_dict("records")
    print(f"\nScope: TOP 5 strategies ({len(strategies_to_output)} reports)")
elif EVALUATION_SCOPE == "all":
    if "oos_results_df" not in locals():
        raise ValueError("Section 7 must be run with EVALUATION_SCOPE='all' first")
    strategies_to_output = oos_results_df.to_dict("records")
    print(f"\nScope: ALL strategies ({len(strategies_to_output)} reports)")
else:
    raise ValueError(
        f"Invalid EVALUATION_SCOPE: {EVALUATION_SCOPE}. Use 'optimal', 'top5', or 'all'"
    )

# Prepare common data (RF and MKTRF)
rf_oos_report = (
    df_oos.reset_index()[["date", "rf"]]
    .dropna()
    .groupby("date", as_index=True)["rf"]
    .mean()
    .astype(float)
    .sort_index()
)

mktrf_oos_report = (
    df_oos.reset_index()[["date", "mktrf"]]
    .dropna()
    .groupby("date", as_index=True)["mktrf"]
    .mean()
    .astype(float)
    .sort_index()
)

generated_reports = []

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
reports_dir = f"outputs/run_{now}"
ensure_dir(reports_dir)

for idx_out, strategy_dict in enumerate(strategies_to_output):
    strat_scoring = strategy_dict["scoring_method"]
    strat_allocation = strategy_dict["allocation_strategy"]
    strat_returns = strategy_dict["portfolio_returns"]
    allocation_strategy_name = ALLOCATION_STRATEGY_NAMES.get(strat_allocation, strat_allocation)

    # Convert returns to equity curve
    equity_curve = (1 + strat_returns).cumprod()

    # File names
    output_file_daily = f"{reports_dir}/oos_{strat_scoring}_{allocation_strategy_name}.html"
    output_file_monthly = f"{reports_dir}/oos_{strat_scoring}_{allocation_strategy_name}_monthly.html"

    # Report titles
    start_date = strat_returns.index.min().date()
    end_date = strat_returns.index.max().date()
    title_daily = (
        f"OOS (Daily): {strat_scoring}+{allocation_strategy_name} ({start_date} to {end_date})"
    )
    title_monthly = (
        f"OOS (Monthly): {strat_scoring}+{allocation_strategy_name} ({start_date} to {end_date})"
    )

    # Generate reports (suppress output)
    make_qs_report_from_equity(
        equity_series=equity_curve,
        rf_series=rf_oos_report,
        mktrf_series=mktrf_oos_report,
        title=title_daily,
        out_path=output_file_daily,
        freq="D",
    )

    make_qs_report_from_equity(
        equity_series=equity_curve,
        rf_series=rf_oos_report,
        mktrf_series=mktrf_oos_report,
        title=title_monthly,
        out_path=output_file_monthly,
        freq="M",
    )

    generated_reports.append(
        {
            "strategy": f"{strat_scoring}+{allocation_strategy_name}",
            "daily": output_file_daily,
            "monthly": output_file_monthly,
        }
    )

# Summary
print("=" * 80)
print("REPORTS GENERATED")
print("=" * 80)

for idx_rep, rep in enumerate(generated_reports, 1):
    print(f"\n{idx_rep}. {rep['strategy']}")
    print(f"   Daily:   {rep['daily']}")
    print(f"   Monthly: {rep['monthly']}")

print(
    f"\nTotal: {len(generated_reports)} strategies × 2 frequencies = {len(generated_reports)*2} reports"
)
print("=" * 80)
