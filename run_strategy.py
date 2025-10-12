import os
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from functions.helpers.allocation_strategies import apply_allocation_strategy
from functions.helpers.data_extraction import wrds_extract_raw
from functions.helpers.output_generation import (
    generate_oos_report,
    generate_oos_report_monthly,
)
from functions.helpers.portfolio_backtest import (
    backtest_strategy,
    calculate_equity_curve,
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
    start="2016-01-01",
    end="2021-01-01",
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
        "AAPL",
        "NVDA",
        "MSFT",
        "AMZN",
        "TSLA",
        "GOOGL",
        "LLY",
        "WMT",
        "JPM",
        "BRK-B",
        #'V', 'MA', 'XOM', 'ORCL', 'UNH', 'COST', 'PG', 'HD', 'NFLX',
        #'JNJ', 'BAC', 'CRM', 'QQQ', 'ABBV', 'KO', 'CVX', 'TMUS', 'MRK', 'CSCO',
        #'WFC', 'ACN', 'NOW', 'TSM', 'AXP', 'PEP', 'MCD', 'IBM', 'MS', 'DIS',
        #'TMO', 'ABT', 'AMD', 'ADBE', 'PM', 'ISRG', 'GE', 'GS', 'INTU', 'CAT',
        #'TXN', 'QCOM', 'RY', 'VZ', 'DHR', 'BKNG', 'T', 'BLK', 'SPGI',
        #'RTX', 'PFE', 'NEE', 'HON', 'CMCSA', 'PGR', 'AMGN', 'LOW', 'ANET', 'UNP',
        #'SYK', 'TJX', 'C', 'BA', 'SCHW', 'BSX', 'KKR', 'ETN',
        #'COP', 'BX', 'PANW', 'ADP'
    ],
)

# =============================================================
# 2. Train/test split & rolling CV split
# =============================================================

# Execute the split
random_state = 42
split_pct = 0.80
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

# 3.2 Hyperparameter tuning: logistic regression + ElasticNet

# ElasticNet Hyperparameters and Pipeline Configuration (prints converted to comments)
# HYPERPARAMETER TUNING CONFIGURATION
# Toggle between hardcoded hyperparameters (fast) and grid search tuning (slow)
TUNE_HYPERPARAMETERS = False  # Set to True to enable full grid search, False for hardcoded values

# Hardcoded hyperparameters (used when TUNE_HYPERPARAMETERS = False)
ELASTICNET_C = 0.1  # Inverse of regularization strength
ELASTICNET_L1_RATIO = 0.7  # L1/L2 tradeoff (0 ridge, 1 lasso)

# Grid search ranges (used when TUNE_HYPERPARAMETERS = True)
param_grid = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__l1_ratio": [0.3, 0.5, 0.7],
}

# Preprocessing: Standardize features
ct = ColumnTransformer(
    [
        (
            "num",  # numerical
            # scales each feature so ElasticNet penalty treats them similarly (avoid leakage)
            StandardScaler(with_mean=True),
            num_pred_cols,  # only numerical columns
        )
    ],
    remainder="drop",  # columns not listed are dropped
    sparse_threshold=0.0,  # force dense for feature importance
)

# Classifier with ElasticNet regularization
clf = LogisticRegression(
    penalty="elasticnet",  # Use ElasticNet (L1 + L2)
    solver="saga",  # Only solver supporting elasticnet
    l1_ratio=ELASTICNET_L1_RATIO,  # Mix of L1 and L2
    C=ELASTICNET_C,  # Regularization strength
    max_iter=5000,  # More iterations for convergence
    tol=1e-4,
    random_state=random_state,
    n_jobs=-1,  # Use all CPUs
)

# Pipeline: preprocessing to classification
pipe = Pipeline([("prep", ct), ("clf", clf)])

# 3.3 Rolling training & feature selection per fold (Model Selection and Cross-Validation)

print("Rolling Window Training With Feature Selection (In-Sample Only)")

pred_prob_up_new = pd.Series(index=df.index, dtype=float)
pred_prob_down_new = pd.Series(index=df.index, dtype=float)
pred_score_new = pd.Series(index=df.index, dtype=float)
pred_class_new = pd.Series(index=df.index, dtype=int)
used_mask_new = pd.Series(False, index=df.index)

# Track feature selection across windows
feature_selection_history = []

window_num = 0
start_pos = 0

while start_pos + ins_training_window_size + ins_validation_window_size <= len(ins_dates):
    window_num += 1

    train_dates = ins_dates[start_pos : start_pos + ins_training_window_size]
    valid_dates = ins_dates[
        start_pos
        + ins_training_window_size : start_pos
        + ins_training_window_size
        + ins_validation_window_size
    ]

    print(f"\nWindow {window_num}: {train_dates.min().date()} to {valid_dates.max().date()}")

    start_pos += step_size

    idx_tr = df.index.get_level_values("date").isin(train_dates)
    idx_va = df.index.get_level_values("date").isin(valid_dates)

    X_tr = df.loc[idx_tr, num_pred_cols]
    y_tr = DIR_binary.loc[idx_tr]
    X_va = df.loc[idx_va, num_pred_cols]
    y_va = DIR_binary.loc[idx_va]
    groups_ins = X_tr.index.get_level_values("date")

    # Conditional hyperparameter tuning based on TUNE_HYPERPARAMETERS flag
    if TUNE_HYPERPARAMETERS:
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="neg_log_loss",  # minimize log loss (cross-entropy)
            cv=GroupKFold(n_splits=3),
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_tr, y_tr, groups=groups_ins)
        pipe_best = gs.best_estimator_
        best_C = gs.best_params_["clf__C"]
        best_l1 = gs.best_params_["clf__l1_ratio"]
        cv_score = gs.best_score_
    else:
        # Use hardcoded hyperparameters (faster)
        pipe_best = clone(pipe)  # Create fresh copy to avoid reusing fitted scaler
        pipe_best.fit(X_tr, y_tr)  # noqa
        best_C = ELASTICNET_C
        best_l1 = ELASTICNET_L1_RATIO
        cv_score = None  # No CV score when not tuning

    # Validation log loss report
    p_up_val = pipe_best.predict_proba(X_va)[:, 1]
    val_log_loss = log_loss(y_va, pipe_best.predict_proba(X_va))
    print(f"   Best params: C={best_C}, l1={best_l1}")
    print(f"   Validation Log Loss: {val_log_loss:.4f}")
    if cv_score is not None:
        print(f"   Train (CV) Log Loss: {-cv_score:.4f}")

    # Feature Selection Analysis (from tuned model)
    coef = pipe_best.named_steps["clf"].coef_[0]

    feature_importance = pd.DataFrame(
        {"feature": num_pred_cols, "coefficient": coef, "abs_coefficient": np.abs(coef)}
    )

    ZERO_THRESHOLD = 1e-5
    selected_features = feature_importance[feature_importance["abs_coefficient"] > ZERO_THRESHOLD]
    n_selected = len(selected_features)
    pct_selected = (n_selected / len(num_pred_cols)) * 100

    top_features = selected_features.nlargest(10, "abs_coefficient")

    feature_selection_history.append(
        {
            "window": window_num,
            "n_features_selected": n_selected,
            "pct_selected": pct_selected,
            "selected_features": selected_features["feature"].tolist(),
            "top_5_features": top_features.head(5)["feature"].tolist(),
            "train_start": train_dates.min(),
            "valid_end": valid_dates.max(),
        }
    )

    # Predictions from the tuned model
    proba = pipe_best.predict_proba(X_va)
    p_down = proba[:, 0]
    p_up = proba[:, 1]
    score = 2 * p_up - 1
    yhat = (p_up > 0.5).astype(int)

    # Store predictions
    pred_prob_up_new.loc[idx_va] = p_up
    pred_prob_down_new.loc[idx_va] = p_down
    pred_score_new.loc[idx_va] = score
    pred_class_new.loc[idx_va] = yhat
    used_mask_new.loc[idx_va] = True

    # Report
    accuracy = (yhat == y_va).mean()  # noqa
    print(f"Training: {len(X_tr):,} samples")
    print(f"Validation: {len(X_va):,} samples, Accuracy: {accuracy:.1%}")
    print("\nFeature Selection:")
    print(f"   Selected: {n_selected}/{len(num_pred_cols)} features ({pct_selected:.1f}%)")
    print("   Top 5 features by importance:")
    for i, row in top_features.head(5).iterrows():
        print(f"      {i + 1}. {row['feature']}: {row['coefficient']:+.4f}")

print(f"\nTraining complete: {window_num} windows processed")
print(f"Total validated: {used_mask_new.sum():,} / {len(df):,}")

# Update global predictions with new ones
pred_prob_up = pred_prob_up_new
pred_prob_down = pred_prob_down_new
pred_score = pred_score_new
pred_class = pred_class_new
used_mask = used_mask_new

# 3.4 Feature selection across all folds

print("Feature Selection For Final Model (Statistical Stability Testing)")
# Extract selection information from all windows
n_windows = len(feature_selection_history)
feature_counts = {feat: 0 for feat in num_pred_cols}

# Count how many windows each feature was selected in
for window_info in feature_selection_history:
    selected_features = window_info["selected_features"]
    for feat in selected_features:
        feature_counts[feat] += 1

# Calculate frequency and statistical significance
feature_freq = pd.DataFrame(
    [
        {
            "feature": feat,
            "count": count,
            "frequency": count / n_windows if n_windows > 0 else 0.0,
            "p_value": (
                binomtest(count, n_windows, p=0.5, alternative="greater").pvalue
                if n_windows > 0
                else 1.0
            ),
        }
        for feat, count in feature_counts.items()
    ]
).sort_values("frequency", ascending=False)

# Statistical threshold: Features must be significantly better than random
SIGNIFICANCE_LEVEL = 0.05  # Standard 5% significance level
# For multiple testing correction, Bonferroni could be used (commented out)

selected_features_mask = feature_freq["p_value"] < SIGNIFICANCE_LEVEL
final_feature_list = feature_freq.loc[selected_features_mask, "feature"].tolist()

print("\n  Selection Criterion:")
print(f"   Statistical significance: alpha = {SIGNIFICANCE_LEVEL}")
print("   Test: Binomial test against H0: feature selected randomly (p=0.5)")

# Calculate minimum required appearances for significance
if n_windows > 0:
    min_appearances_for_sig = min(
        [
            i
            for i in range(n_windows + 1)
            if binomtest(i, n_windows, 0.5, alternative="greater").pvalue < SIGNIFICANCE_LEVEL
        ]
    )
else:
    min_appearances_for_sig = 0

print("\n  Statistical Requirement:")
print(
    f"   For {n_windows} windows, need >={min_appearances_for_sig}/{n_windows} appearances "
    f"(>={min_appearances_for_sig/n_windows:.0%} if n_windows>0 else 0%)"
)
if n_windows > 0:
    print(
        f"   Note: 50% threshold ({n_windows//2}/{n_windows}) has p-value: "
        f"{binomtest(n_windows//2, n_windows, 0.5, alternative='greater').pvalue:.3f}"
    )

print("\n  Results:")
print(f"   Features selected: {len(final_feature_list)} / {len(num_pred_cols)}")
print(f"   Features removed:  {len(num_pred_cols) - len(final_feature_list)}")
print(f"   Reduction: {(1 - len(final_feature_list) / len(num_pred_cols)) * 100:.1f}%")

# Display selected features
print(f"\n  Selected Features ({len(final_feature_list)}) - Statistically Significant:")
print(f"    {'Feature':<30} {'Frequency':>10} {'Count':>8} {'P-value':>10} {'Sig'}")
print("    " + "=" * 70)

selected_features_df = feature_freq[selected_features_mask].copy()
for idx, row in selected_features_df.iterrows():
    feat = row["feature"]
    freq = row["frequency"]
    count = row["count"]
    p_val = row["p_value"]
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    bar = "█" * int(freq * 20)
    print(f"    {feat:<30} {freq:>6.1%} ({count:>2}/{n_windows})  p={p_val:>6.4f} {sig:>5}  {bar}")

# Display removed features (top 15 by frequency, but not statistically significant)
removed_features_df = feature_freq[~selected_features_mask].copy()

if len(removed_features_df) > 0:
    print(f"\n  Removed Features ({len(removed_features_df)}) - Not Statistically Significant:")
    print("     (showing top 15 by frequency)")
    print(f"    {'Feature':<30} {'Frequency':>10} {'Count':>8} {'P-value':>10}")
    print("    " + "=" * 70)

    for idx, row in removed_features_df.head(15).iterrows():
        feat = row["feature"]
        freq = row["frequency"]
        count = row["count"]
        p_val = row["p_value"]
        bar = "░" * int(freq * 20)
        print(f"    {feat:<30} {freq:>6.1%} ({count:>2}/{n_windows})  p={p_val:>6.4f} n.s.  {bar}")


# Feature categories breakdown
def categorize_feature(feat):
    """Categorize feature by type"""
    if feat.startswith("ti_"):
        return "Technical Indicator applied to ticker"
    if feat.startswith("comm_"):
        if "ti_" in feat:
            return "Technical Indicator applied to common feature"
        else:
            return "Common feature"
    if feat.startswith("ti_"):
        return "Technical Indicator applied to ticker"
    elif feat in ["mktrf", "smb", "hml", "rf", "umd"]:
        return "Fama-French Factors"
    elif feat.startswith("cons_") or feat.startswith("n_"):
        return "IBES Consensus"
    elif feat.startswith("adjclose_lag"):
        return "Price Lags"
    elif feat in ["adj_mktcap", "vol", "retx"]:
        return "Market Data"
    else:
        return "Other"


selected_features_df["category"] = selected_features_df["feature"].apply(categorize_feature)
category_counts = selected_features_df["category"].value_counts()

print("\n  Selected Features By Category:")
for category, count in category_counts.items():
    pct = count / len(selected_features_df) * 100 if len(selected_features_df) else 0.0
    print(f"    {category:<30} {count:>2} features ({pct:>5.1f}%)")

# 3.5 Train the final in-sample model on all in-sample data

# Global Hyperparameter Search (in-sample, using selected features) — informational prints converted to comments
# Using Ridge (L2-only) for final model after stability-based feature selection

# Build in-sample slice with final features
ins_mask = df.index.get_level_values("date").isin(ins_dates)
X_ins = df.loc[ins_mask, final_feature_list]
y_ins = DIR_binary.loc[ins_mask]

# Group by date so folds split by day (avoid same-day leakage across stocks)
groups_ins = df.loc[ins_mask].index.get_level_values("date")

# Rebuild pipeline constrained to the selected features
ct_final_cv = ColumnTransformer(
    [("num", StandardScaler(with_mean=True), final_feature_list)],
    remainder="drop",
    sparse_threshold=0.0,
)
clf_final_cv = LogisticRegression(
    penalty="l2",  # Ridge regularization only
    solver="lbfgs",
    max_iter=5000,
    tol=1e-4,
    random_state=random_state,
    n_jobs=-1,
)
pipe_final_cv = Pipeline([("prep", ct_final_cv), ("clf", clf_final_cv)])

# Conditional global hyperparameter tuning based on TUNE_HYPERPARAMETERS flag
if TUNE_HYPERPARAMETERS:
    param_grid_global = {
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
    }
    cv_global = GroupKFold(n_splits=5)
    gs_global = GridSearchCV(
        estimator=pipe_final_cv,
        param_grid=param_grid_global,
        scoring="neg_log_loss",
        cv=cv_global,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs_global.fit(X_ins, y_ins, groups=groups_ins)
    best_C = gs_global.best_params_["clf__C"]
    # Print summary table safely (no display())
    results_df = pd.DataFrame(gs_global.cv_results_)
    table = (
        results_df[["param_clf__C", "mean_test_score", "std_test_score"]]
        .sort_values("mean_test_score", ascending=False)
        .to_string(index=False)
    )
    print("\nRidge C grid-search summary (higher mean_test_score is better neg_log_loss):")
    print(table)
else:
    # Hardcoded hyperparameter (faster)
    RIDGE_C = 0.1  # Moderate regularization
    best_C = RIDGE_C

# Lock in the global best hyperparameters and train final in-sample model
clf_final = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    C=best_C,
    max_iter=5000,
    tol=1e-4,
    random_state=random_state,
    n_jobs=-1,
)
ct_final = ColumnTransformer(
    [("num", StandardScaler(with_mean=True), final_feature_list)],
    remainder="drop",
    sparse_threshold=0.0,
)
pipe_final = Pipeline([("prep", ct_final), ("clf", clf_final)])
pipe_final.fit(X_ins, y_ins)

# Report final model
final_coef = pipe_final.named_steps["clf"].coef_[0]
non_zero = (np.abs(final_coef) > 1e-5).sum()
print("\nFinal Ridge Model (classification):")
print(f"  Regularization: C={best_C} (lower = stronger)")
print(f"  Total features: {len(final_coef)}")
print(f"  Non-zero coeffs: {non_zero} / {len(final_coef)}")

# Check probability distribution on training data
train_probs = pipe_final.predict_proba(X_ins)[:, 1]
print("\nIn-Sample Probability Distribution:")
print(f"  Mean:   {train_probs.mean():.4f}")
print(f"  Std:    {train_probs.std():.4f}")
print(f"  Range:  [{train_probs.min():.4f}, {train_probs.max():.4f}]")
extreme_pct = ((train_probs < 0.05) | (train_probs > 0.95)).mean() * 100
print(f"  Extreme predictions (<0.05 or >0.95): {extreme_pct:.1f}%")
if train_probs.max() > 0.99 or train_probs.min() < 0.01:
    print("  WARNING: Very extreme probabilities detected (check OOS performance).")
elif extreme_pct > 20:
    print(f"  Note: {extreme_pct:.1f}% of predictions are very confident (<0.05 or >0.95)")

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

# 4.2 Hyperparameter tuning: linear regression + ElasticNet

# Linear regression uses ElasticNet (L1 + L2 regularization)
# Use same hyperparameter tuning toggle as logistic regression

# Hardcoded hyperparameters for linear regression (ElasticNet)
LINEAR_ALPHA = 0.001
LINEAR_L1_RATIO = 0.5

# Grid search ranges (used when TUNE_HYPERPARAMETERS = True)
param_grid_linear = {
    "clf__alpha": [0.0001, 0.001, 0.01, 0.1],
    "clf__l1_ratio": [0.3, 0.5, 0.7, 0.9],
}

# Preprocessing: Standardize features (same as logistic)
ct_linear = ColumnTransformer(
    [("num", StandardScaler(with_mean=True), num_pred_cols)], remainder="drop", sparse_threshold=0.0
)

# ElasticNet regression with L1 + L2 regularization
clf_linear = ElasticNet(
    alpha=LINEAR_ALPHA, l1_ratio=LINEAR_L1_RATIO, max_iter=5000, tol=1e-4, random_state=random_state
)

# Pipeline: preprocessing to regression
pipe_linear = Pipeline([("prep", ct_linear), ("clf", clf_linear)])

# 4.3 Rolling training & feature selection per fold (Model Selection and Cross-Validation)

print("Rolling Window Training for Linear Regression (In-Sample Only)")

pred_return_linear = pd.Series(index=df.index, dtype=float)
used_mask_linear = pd.Series(False, index=df.index)

# Track feature selection across windows
feature_selection_history_linear = []

window_num = 0
start_pos = 0

while start_pos + ins_training_window_size + ins_validation_window_size <= len(ins_dates):
    window_num += 1
    window_start_time = time.time()

    train_dates = ins_dates[start_pos : start_pos + ins_training_window_size]
    valid_dates = ins_dates[
        start_pos
        + ins_training_window_size : start_pos
        + ins_training_window_size
        + ins_validation_window_size
    ]

    print(f"\n{'='*70}")
    print(f"Window {window_num}: {train_dates.min().date()} to {valid_dates.max().date()}")

    start_pos += step_size

    idx_tr = df.index.get_level_values("date").isin(train_dates)
    idx_va = df.index.get_level_values("date").isin(valid_dates)

    X_tr = df.loc[idx_tr, num_pred_cols]
    y_tr = y_continuous.loc[idx_tr]
    X_va = df.loc[idx_va, num_pred_cols]
    y_va = y_continuous.loc[idx_va]
    groups_ins = X_tr.index.get_level_values("date")

    print(f"Training samples: {len(X_tr):,} | Validation samples: {len(X_va):,}")

    # Conditional hyperparameter tuning based on TUNE_HYPERPARAMETERS flag
    fit_start_time = time.time()
    if TUNE_HYPERPARAMETERS:
        # Grid search with cross-validation
        gs_linear = GridSearchCV(
            estimator=pipe_linear,
            param_grid=param_grid_linear,
            scoring="neg_mean_squared_error",  # maximize -MSE == minimize MSE
            cv=GroupKFold(n_splits=3),
            n_jobs=-1,
            refit=True,
        )
        gs_linear.fit(X_tr, y_tr, groups=groups_ins)
        pipe_best_linear = gs_linear.best_estimator_
        best_alpha = gs_linear.best_params_["clf__alpha"]
        best_l1_ratio = gs_linear.best_params_["clf__l1_ratio"]
        cv_score = gs_linear.best_score_
    else:
        # Use hardcoded hyperparameters (faster) - create fresh clone
        pipe_best_linear = clone(pipe_linear)
        pipe_best_linear.fit(X_tr, y_tr)  # noqa
        best_alpha = LINEAR_ALPHA
        best_l1_ratio = LINEAR_L1_RATIO
        cv_score = None

    fit_time = time.time() - fit_start_time
    print(f"   Fitting time: {fit_time:.2f} seconds")

    # Validation RMSE/R2 report
    y_pred_val = pipe_best_linear.predict(X_va)
    rmse_val = np.sqrt(mean_squared_error(y_va, y_pred_val))
    r2_val = r2_score(y_va, y_pred_val)

    print(f"   Best params: alpha={best_alpha}, l1_ratio={best_l1_ratio}")
    print(f"   Validation RMSE: {rmse_val:.6f}")
    print(f"   Validation R²: {r2_val:.4f}")
    if cv_score is not None:
        print(f"   train(CV) RMSE: {np.sqrt(-cv_score):.6f}")

    # Feature Selection Analysis (from tuned model)
    coef = pipe_best_linear.named_steps["clf"].coef_

    feature_importance = pd.DataFrame(
        {"feature": num_pred_cols, "coefficient": coef, "abs_coefficient": np.abs(coef)}
    )

    ZERO_THRESHOLD = 1e-5
    selected_features = feature_importance[feature_importance["abs_coefficient"] > ZERO_THRESHOLD]
    n_selected = len(selected_features)
    pct_selected = (n_selected / len(num_pred_cols)) * 100

    top_features = selected_features.nlargest(10, "abs_coefficient")

    feature_selection_history_linear.append(
        {
            "window": window_num,
            "n_features_selected": n_selected,
            "pct_selected": pct_selected,
            "selected_features": selected_features["feature"].tolist(),
            "top_5_features": top_features.head(5)["feature"].tolist(),
            "train_start": train_dates.min(),
            "valid_end": valid_dates.max(),
        }
    )

    # Store predictions
    pred_return_linear.loc[idx_va] = y_pred_val
    used_mask_linear.loc[idx_va] = True

    # Report
    print("\nFeature Selection:")
    print(f"   Selected: {n_selected}/{len(num_pred_cols)} features ({pct_selected:.1f}%)")
    print("   Top 5 features by importance:")
    for i, row in top_features.head(5).iterrows():
        print(f"      {i + 1}. {row['feature']}: {row['coefficient']:+.6f}")

    window_time = time.time() - window_start_time
    print(
        f"\nWindow {window_num} total time: {window_time:.2f} seconds ({window_time/60:.2f} minutes)"
    )

print(f"\nLinear Regression Training complete: {window_num} windows processed")
print(f"Total validated: {used_mask_linear.sum():,} / {len(df):,}")

# 4.4 Feature selection across all folds

print("Feature Selection For Linear Regression Model (Statistical Stability Testing)")

# Extract selection information from all windows
n_windows_linear = len(feature_selection_history_linear)
feature_counts_linear = {feat: 0 for feat in num_pred_cols}

# Count how many windows each feature was selected in
for window_info in feature_selection_history_linear:
    selected_features = window_info["selected_features"]
    for feat in selected_features:
        feature_counts_linear[feat] += 1

# Calculate frequency and statistical significance
feature_freq_linear = pd.DataFrame(
    [
        {
            "feature": feat,
            "count": count,
            "frequency": count / n_windows_linear if n_windows_linear > 0 else 0.0,
            "p_value": (
                binomtest(count, n_windows_linear, p=0.5, alternative="greater").pvalue
                if n_windows_linear > 0
                else 1.0
            ),
        }
        for feat, count in feature_counts_linear.items()
    ]
).sort_values("frequency", ascending=False)

# Statistical threshold
SIGNIFICANCE_LEVEL_LINEAR = 0.05

selected_features_mask_linear = feature_freq_linear["p_value"] < SIGNIFICANCE_LEVEL_LINEAR
final_feature_list_linear = feature_freq_linear.loc[
    selected_features_mask_linear, "feature"
].tolist()

print("\n  Selection Criterion:")
print(f"   Statistical significance: alpha = {SIGNIFICANCE_LEVEL_LINEAR}")
print("   Test: Binomial test against H0: feature selected randomly (p=0.5)")

# Minimum required appearances
if n_windows_linear > 0:
    min_appearances_for_sig_linear = min(
        [
            i
            for i in range(n_windows_linear + 1)
            if binomtest(i, n_windows_linear, 0.5, alternative="greater").pvalue
            < SIGNIFICANCE_LEVEL_LINEAR
        ]
    )
else:
    min_appearances_for_sig_linear = 0

print("\n  Statistical Requirement:")
print(
    f"   For {n_windows_linear} windows, need >={min_appearances_for_sig_linear}/{n_windows_linear} appearances "
    f"(>={min_appearances_for_sig_linear/n_windows_linear:.0%} if n_windows_linear>0 else 0%)"
)
if n_windows_linear > 0:
    print(
        f"   Note: 50% threshold ({n_windows_linear//2}/{n_windows_linear}) has p-value: "
        f"{binomtest(n_windows_linear//2, n_windows_linear, 0.5, alternative='greater').pvalue:.3f}"
    )

print("\n  Results:")
print(f"   Features selected: {len(final_feature_list_linear)} / {len(num_pred_cols)}")
print(f"   Features removed:  {len(num_pred_cols) - len(final_feature_list_linear)}")
print(f"   Reduction: {(1 - len(final_feature_list_linear) / len(num_pred_cols)) * 100:.1f}%")

# Safety check: ensure at least some features were selected
if len(final_feature_list_linear) == 0:
    print("\n  WARNING: No features selected with current statistical threshold!")
    raise ValueError(
        "No features selected for linear regression model. Adjust hyperparameters or significance level."
    )

# Display selected features
print(f"\n  Selected Features ({len(final_feature_list_linear)}) - Statistically Significant:")
print(f"    {'Feature':<30} {'Frequency':>10} {'Count':>8} {'P-value':>10} {'Sig'}")
print("    " + "=" * 70)

selected_features_df_linear = feature_freq_linear[selected_features_mask_linear].copy()
for idx, row in selected_features_df_linear.iterrows():
    feat = row["feature"]
    freq = row["frequency"]
    count = row["count"]
    p_val = row["p_value"]
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    bar = "█" * int(freq * 20)
    print(
        f"    {feat:<30} {freq:>6.1%} ({count:>2}/{n_windows_linear})  p={p_val:>6.4f} {sig:>5}  {bar}"
    )

# Compare with logistic regression features
print("\n  Comparison with Logistic Regression Features:")
logistic_features = set(final_feature_list)
linear_features = set(final_feature_list_linear)
common_features = logistic_features.intersection(linear_features)
logistic_only = logistic_features - linear_features
linear_only = linear_features - logistic_features
print(f"    Common to both models:    {len(common_features)} features")
print(f"    Logistic only:            {len(logistic_only)} features")
print(f"    Linear only:              {len(linear_only)} features")
if 0 < len(logistic_only) <= 10:
    print(f"    Logistic-only features: {', '.join(list(logistic_only)[:10])}")
if 0 < len(linear_only) <= 10:
    print(f"    Linear-only features: {', '.join(list(linear_only)[:10])}")

# 4.5 Train the final in-sample model on all in-sample data

# Using Ridge (L2-only) for final model after stability-based feature selection

# Build in-sample slice with final features
ins_mask_linear = df.index.get_level_values("date").isin(ins_dates)
X_ins_linear = df.loc[ins_mask_linear, final_feature_list_linear]
y_ins_linear = y_continuous.loc[ins_mask_linear]

# Group by date so folds split by day (avoid same-day leakage across stocks)
groups_ins_linear = df.loc[ins_mask_linear].index.get_level_values("date")

# Rebuild pipeline constrained to the selected features
ct_final_linear = ColumnTransformer(
    [("num", StandardScaler(with_mean=True), final_feature_list_linear)],
    remainder="drop",
    sparse_threshold=0.0,
)
clf_final_linear_cv = Ridge(max_iter=5000, tol=1e-4, random_state=random_state)
pipe_final_linear_cv = Pipeline([("prep", ct_final_linear), ("clf", clf_final_linear_cv)])

# Conditional global hyperparameter tuning
if TUNE_HYPERPARAMETERS:
    param_grid_linear_global = {
        "clf__alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    }
    cv_global_linear = GroupKFold(n_splits=5)
    gs_global_linear = GridSearchCV(
        estimator=pipe_final_linear_cv,
        param_grid=param_grid_linear_global,
        scoring="neg_mean_squared_error",
        cv=cv_global_linear,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs_global_linear.fit(X_ins_linear, y_ins_linear, groups=groups_ins_linear)
    best_alpha_linear = gs_global_linear.best_params_["clf__alpha"]
    # Print summary table safely
    results_df_linear = pd.DataFrame(gs_global_linear.cv_results_)
    table_lin = (
        results_df_linear[["param_clf__alpha", "mean_test_score", "std_test_score"]]
        .sort_values("mean_test_score", ascending=False)
        .to_string(index=False)
    )
    print("\nRidge alpha grid-search summary (higher mean_test_score is better -MSE):")
    print(table_lin)
else:
    RIDGE_ALPHA = 0.1  # Moderate regularization (higher = stronger)
    best_alpha_linear = RIDGE_ALPHA

# Lock in global best and train final in-sample model
clf_final_linear = Ridge(
    alpha=best_alpha_linear, max_iter=5000, tol=1e-4, random_state=random_state
)
ct_final_linear_final = ColumnTransformer(
    [("num", StandardScaler(with_mean=True), final_feature_list_linear)],
    remainder="drop",
    sparse_threshold=0.0,
)
pipe_final_linear = Pipeline([("prep", ct_final_linear_final), ("clf", clf_final_linear)])
pipe_final_linear.fit(X_ins_linear, y_ins_linear)

# Report model performance on in-sample
y_pred_ins = pipe_final_linear.predict(X_ins_linear)
rmse_ins = np.sqrt(mean_squared_error(y_ins_linear, y_pred_ins))
r2_ins = r2_score(y_ins_linear, y_pred_ins)

print("\nFinal Ridge Model Performance (In-Sample, regression):")
print(f"  Regularization: alpha={best_alpha_linear} (higher = stronger)")
print(f"  RMSE: {rmse_ins:.6f}")
print(f"  R²: {r2_ins:.4f}")

# Report final model - Ridge doesn't eliminate
final_coef_linear = pipe_final_linear.named_steps["clf"].coef_
non_zero_linear = (np.abs(final_coef_linear) > 1e-5).sum()
print(f"  Total features: {len(final_feature_list_linear)}")
print(f"  Non-zero coeffs: {non_zero_linear} / {len(final_feature_list_linear)}")

# Prediction distribution
print("\nIn-Sample Prediction Distribution (regression):")
print(f"  Mean:   {y_pred_ins.mean():.6f}")
print(f"  Std:    {y_pred_ins.std():.6f}")
print(f"  Range:  [{y_pred_ins.min():.6f}, {y_pred_ins.max():.6f}]")

# Actual vs Predicted
print("\nActual vs Predicted (regression):")
print(f"  Actual mean:     {y_ins_linear.mean():.6f}")
print(f"  Predicted mean:  {y_pred_ins.mean():.6f}")
print(f"  Difference:      {abs(y_pred_ins.mean() - y_ins_linear.mean()):.6f}")

# =============================================================
# 5. Signal Confirmation & Trading Universe Selection
# =============================================================

# 5.1 Signal Confirmation

print("SIGNAL CONFIRMATION: Ensemble Agreement Filter")

# Configuration thresholds (converted to comments)
# Logistic thresholds: LONG if prob_up > PROB_UP_THRESHOLD_LONG; SHORT if prob_up < 1 - PROB_UP_THRESHOLD_SHORT
PROB_UP_THRESHOLD_LONG = 0.50
PROB_UP_THRESHOLD_SHORT = 0.50

# Linear thresholds (in absolute return terms)
EXPECTED_RETURN_THRESHOLD_LONG = 0.0
EXPECTED_RETURN_THRESHOLD_SHORT = 0.0

# Generate Predictions for ALL Data (In-Sample + Out-Of-Sample)
# Logistic Regression Predictions
X_full_logistic = df[final_feature_list].copy()
prob_up_full = pipe_final.predict_proba(X_full_logistic)[:, 1]
prob_down_full = 1 - prob_up_full
logistic_score_full = 2 * prob_up_full - 1

# Linear Regression Predictions
X_full_linear = df[final_feature_list_linear].copy()
expected_return_full = pipe_final_linear.predict(X_full_linear)

# Store predictions in DataFrame
df_signals = df.copy()
df_signals["prob_up"] = prob_up_full
df_signals["prob_down"] = prob_down_full
df_signals["logistic_score"] = logistic_score_full
df_signals["expected_return"] = expected_return_full

print(f"\nPredictions generated for {len(df_signals):,} observations")
print(f"   Logistic: prob(up) in [{prob_up_full.min():.4f}, {prob_up_full.max():.4f}]")
print(
    f"   Linear:   E[R]     in [{expected_return_full.min():.6f}, {expected_return_full.max():.6f}]"
)

# Define Trading Signals from Each Model
# Logistic Regression Signals
df_signals["logistic_signal_long"] = df_signals["prob_up"] > PROB_UP_THRESHOLD_LONG
df_signals["logistic_signal_short"] = df_signals["prob_up"] < (1 - PROB_UP_THRESHOLD_SHORT)
df_signals["logistic_signal_neutral"] = ~(
    df_signals["logistic_signal_long"] | df_signals["logistic_signal_short"]
)

# Linear Regression Signals
df_signals["linear_signal_long"] = df_signals["expected_return"] > EXPECTED_RETURN_THRESHOLD_LONG
df_signals["linear_signal_short"] = df_signals["expected_return"] < -EXPECTED_RETURN_THRESHOLD_SHORT
df_signals["linear_signal_neutral"] = ~(
    df_signals["linear_signal_long"] | df_signals["linear_signal_short"]
)

print("\nLogistic Regression Signals:")
print(
    f"  Long:    {df_signals['logistic_signal_long'].sum():>6,} ({df_signals['logistic_signal_long'].mean()*100:>5.1f}%)"
)
print(
    f"  Short:   {df_signals['logistic_signal_short'].sum():>6,} ({df_signals['logistic_signal_short'].mean()*100:>5.1f}%)"
)
print(
    f"  Neutral: {df_signals['logistic_signal_neutral'].sum():>6,} ({df_signals['logistic_signal_neutral'].mean()*100:>5.1f}%)"
)

print("\nLinear Regression Signals:")
print(
    f"  Long:    {df_signals['linear_signal_long'].sum():>6,} ({df_signals['linear_signal_long'].mean()*100:>5.1f}%)"
)
print(
    f"  Short:   {df_signals['linear_signal_short'].sum():>6,} ({df_signals['linear_signal_short'].mean()*100:>5.1f}%)"
)
print(
    f"  Neutral: {df_signals['linear_signal_neutral'].sum():>6,} ({df_signals['linear_signal_neutral'].mean()*100:>5.1f}%)"
)

# Ensemble Agreement Analysis
print("Ensemble Agreement (Both Models Must Agree)")

# Agreement on LONG: Both models say LONG
df_signals["agreed_long"] = df_signals["logistic_signal_long"] & df_signals["linear_signal_long"]

# Agreement on SHORT: Both models say SHORT
df_signals["agreed_short"] = df_signals["logistic_signal_short"] & df_signals["linear_signal_short"]

# Total agreement (either direction)
df_signals["agreed_any"] = df_signals["agreed_long"] | df_signals["agreed_short"]

# Disagreement: Models give conflicting signals
df_signals["disagreed"] = ~df_signals["agreed_any"]

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

# 5.2 Trading Universe Selection

print("TRADING UNIVERSE SELECTION: Filtering & Scoring")

# Create Ensemble Score for Ranking
# Strategy: Use logistic score for ranking, but set to 0 where models disagree
df_signals["ensemble_score"] = df_signals["logistic_score"].copy()
df_signals.loc[df_signals["disagreed"], "ensemble_score"] = 0.0

print("\n Ensemble Score Created:")
print(f"   Non-zero scores (tradeable): {(df_signals['ensemble_score'] != 0).sum():>6,}")  # noqa
print(f"   Zero scores (filtered out):  {(df_signals['ensemble_score'] == 0).sum():>6,}")  # noqa
print(
    f"   Score range: [{df_signals['ensemble_score'].min():.4f}, {df_signals['ensemble_score'].max():.4f}]"
)

print("\n" + "=" * 80)
print("Trading Universe Breakdown by Direction")
print("=" * 80)

# Create separate universes for long, short, and excluded
long_universe = df_signals[df_signals["agreed_long"]].copy()
short_universe = df_signals[df_signals["agreed_short"]].copy()
excluded_universe = df_signals[df_signals["disagreed"]].copy()

print(f"\n LONG Universe:  {len(long_universe):>6,} observations")
if len(long_universe) > 0:
    print(f"   Mean prob(up):       {long_universe['prob_up'].mean():.4f}")
    print(f"   Mean E[R]:           {long_universe['expected_return'].mean():.6f}")
    print(f"   Mean ensemble score: {long_universe['ensemble_score'].mean():.4f}")
else:
    print("   (No long candidates)")

print(f"\n SHORT Universe: {len(short_universe):>6,} observations")
if len(short_universe) > 0:
    print(f"   Mean prob(up):       {short_universe['prob_up'].mean():.4f}")
    print(f"   Mean E[R]:           {short_universe['expected_return'].mean():.6f}")
    print(f"   Mean ensemble score: {short_universe['ensemble_score'].mean():.4f}")
else:
    print("   (No short candidates)")

print(f"\n EXCLUDED:       {len(excluded_universe):>6,} observations (model disagreement)")

print("\n" + "=" * 80)
print("Trading Universe Summary")
print("=" * 80)

total_obs = len(df_signals)
tradeable_obs = len(long_universe) + len(short_universe)
filtered_obs = len(excluded_universe)

print(f"\n  Total observations:      {total_obs:>7,}")
print(f"  Tradeable (agreed):      {tradeable_obs:>7,} ({tradeable_obs/total_obs*100:>5.1f}%)")
print(f"  Filtered out (disagreed): {filtered_obs:>7,} ({filtered_obs/total_obs*100:>5.1f}%)")

# Variables created for downstream use:
# - df_signals:       Full DataFrame with predictions, signals, and ensemble_score
# - long_universe:    Filtered DataFrame for long candidates
# - short_universe:   Filtered DataFrame for short candidates
# - excluded_universe: Filtered DataFrame for disagreements (not traded)

# =============================================================
# 6. Allocation Strategy & Portfolio Construction
# =============================================================

# 6.1 Scoring Methods (configuration prints converted to comments)

# SCORING CONFIGURATION
SCORING_METHOD = "ALL"  # Options: "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "ALL"
USE_WINSORIZATION = True
WINSORIZE_LIMITS = (0.01, 0.99)
USE_CROSS_SECTIONAL_STANDARDIZATION = True
CLIP_PROB_FOR_LOGIT = (0.01, 0.99)
ALPHA_S4 = 0.5
LAMBDA_SOFTMAX = 1.0
USE_STABILITY_FILTER = False
STABILITY_METHOD = "median"  # "median" or "moving_average"
STABILITY_WINDOW = 5

SCORING_DESCRIPTIONS = {
    "S1": "S(1) = p · μ (Product: probability × expected return)",
    "S2": "S(2) = (p - 0.5) · μ (Margin-aware: penalizes coin-flip probs)",
    "S3": "S(3) = logit(p) · μ (Log-odds: stretches confident probs)",
    "S4": "S(4) = α·p̃ + (1-α)·μ̃ (Additive: scale-balanced combo)",
    "S5": "S(5) = p̃ · μ̃ (Multiplicative: scale-balanced combo)",
    "S6": "S(6) = (2p - 1) · |μ| (Expected directional return)",
    "S7": "S(7) = (p · μ) / σ (Risk-adjusted: Sharpe-like signal)",
    "S8": "S(8) = Softmax weights from any score (for portfolio construction)",
}

# Step 1: Preprocessing - Winsorization & Clipping
# Extract raw variables
p_raw = df_signals["prob_up"].copy()
mu_raw = df_signals["expected_return"].copy()

if USE_WINSORIZATION:
    # Winsorize per date to reduce outlier impact
    def winsorize_cross_sectional(series, limits):
        lower, upper = limits
        q_lower = series.quantile(lower)
        q_upper = series.quantile(upper)
        return series.clip(lower=q_lower, upper=q_upper)

    p_winsorized = df_signals.groupby(level="date")["prob_up"].transform(
        lambda x: winsorize_cross_sectional(x, WINSORIZE_LIMITS)
    )
    mu_winsorized = df_signals.groupby(level="date")["expected_return"].transform(
        lambda x: winsorize_cross_sectional(x, WINSORIZE_LIMITS)
    )

    df_signals["p_processed"] = p_winsorized
    df_signals["mu_processed"] = mu_winsorized
else:
    df_signals["p_processed"] = p_raw
    df_signals["mu_processed"] = mu_raw


# Step 2: Cross-Sectional Standardization (Z-scoring per date)
def z_score_cross_sectional(series):
    """Z-score within each date"""
    mean = series.mean()
    std = series.std()
    if std > 0:
        return (series - mean) / std
    return series - mean


if USE_CROSS_SECTIONAL_STANDARDIZATION:
    df_signals["p_tilde"] = df_signals.groupby(level="date")["p_processed"].transform(
        z_score_cross_sectional
    )
    df_signals["mu_tilde"] = df_signals.groupby(level="date")["mu_processed"].transform(
        z_score_cross_sectional
    )
else:
    df_signals["p_tilde"] = df_signals["p_processed"]
    df_signals["mu_tilde"] = df_signals["mu_processed"]

# Step 3: Calculate All 8 Scoring Methods
p = df_signals["p_processed"]
mu = df_signals["mu_processed"]
p_tilde = df_signals["p_tilde"]
mu_tilde = df_signals["mu_tilde"]

# S(1): Product score
df_signals["score_S1"] = p * mu
# S(2): Margin-aware product
df_signals["score_S2"] = (p - 0.5) * mu
# S(3): Log-odds weighting
p_clipped = p.clip(lower=CLIP_PROB_FOR_LOGIT[0], upper=CLIP_PROB_FOR_LOGIT[1])
logit_p = np.log(p_clipped / (1 - p_clipped))
df_signals["score_S3"] = logit_p * mu
# S(4): Additive, scale-balanced combo
df_signals["score_S4"] = ALPHA_S4 * p_tilde + (1 - ALPHA_S4) * mu_tilde
# S(5): Multiplicative, scale-balanced combo
df_signals["score_S5"] = p_tilde * mu_tilde
# S(6): Expected directional return
df_signals["score_S6"] = (2 * p - 1) * mu.abs()
# S(7): Risk-adjusted
df_signals["volatility"] = df_signals.groupby(level="permno")["adj_prc_logret"].transform(
    lambda x: x.rolling(window=20, min_periods=5).std()
)
df_signals["volatility"] = df_signals.groupby(level="date")["volatility"].transform(
    lambda x: x.fillna(x.median())
)
df_signals["score_S7"] = (p * mu) / (df_signals["volatility"] + 1e-8)
# S(8): Softmax base
df_signals["score_S8_base"] = df_signals["score_S1"]

# Step 4: Select Active Scoring Method (or process ALL)
score_column_map = {
    "S1": "score_S1",
    "S2": "score_S2",
    "S3": "score_S3",
    "S4": "score_S4",
    "S5": "score_S5",
    "S6": "score_S6",
    "S7": "score_S7",
    "S8": "score_S8_base",
}

if SCORING_METHOD == "ALL":
    methods_to_process = list(score_column_map.keys())
else:
    if SCORING_METHOD not in score_column_map:
        raise ValueError(f"Unknown scoring method: {SCORING_METHOD}. Use S1-S8 or ALL")
    methods_to_process = [SCORING_METHOD]

all_method_scores = {}

for method in methods_to_process:
    method_base_score = df_signals[score_column_map[method]].copy()

    # Apply softmax per date for S8
    if method == "S8":

        def softmax_per_date(group):
            exp_scores = np.exp(LAMBDA_SOFTMAX * group)
            return exp_scores / exp_scores.sum()

        method_base_score = method_base_score.groupby(level="date").transform(softmax_per_date)

    # Only score observations where models agree
    method_base_score[df_signals["disagreed"]] = 0.0

    all_method_scores[method] = method_base_score

# Set the active base_score
if SCORING_METHOD == "ALL":
    df_signals["base_score"] = all_method_scores["S2"]  # default for downstream
else:
    df_signals["base_score"] = all_method_scores[SCORING_METHOD]

# Step 5: Additional Cross-Sectional Standardization (optional) for base_score
APPLY_SCORE_LEVEL_STANDARDIZATION = SCORING_METHOD in ["S1", "S2", "S3", "S6", "S7"]
if APPLY_SCORE_LEVEL_STANDARDIZATION:

    def cross_sectional_standardize_score(group):
        # Only standardize non-zero scores (where models agree)
        mask = group != 0
        if mask.sum() > 1:  # noqa
            mean = group[mask].mean()
            std = group[mask].std()
            if std > 0:
                group[mask] = (group[mask] - mean) / std
        return group

    df_signals["ranking_score"] = df_signals.groupby(level="date")["base_score"].transform(
        cross_sectional_standardize_score
    )
else:
    df_signals["ranking_score"] = df_signals["base_score"]

# Step 6: Stability Filter (optional)
if USE_STABILITY_FILTER:
    if STABILITY_METHOD == "median":
        df_signals["final_ranking_score"] = df_signals.groupby(level="permno")[
            "ranking_score"
        ].transform(
            lambda x: x.rolling(window=STABILITY_WINDOW, min_periods=1, center=False).median()
        )
    elif STABILITY_METHOD == "moving_average":
        df_signals["final_ranking_score"] = df_signals.groupby(level="permno")[
            "ranking_score"
        ].transform(
            lambda x: x.rolling(window=STABILITY_WINDOW, min_periods=1, center=False).mean()
        )
else:
    df_signals["final_ranking_score"] = df_signals["ranking_score"]

# Step 7: Final Score Summary
final_non_zero = df_signals[df_signals["final_ranking_score"] != 0]["final_ranking_score"]
if len(final_non_zero) > 0:
    print("\nFinal Ranking Score Summary:")
    print(f"  Non-zero scores: {len(final_non_zero):>6,}")
    print(f"  Mean:            {final_non_zero.mean():>10.6f}")
    print(f"  Std:             {final_non_zero.std():>10.6f}")
    print(f"  Min:             {final_non_zero.min():>10.6f}")
    print(f"  Max:             {final_non_zero.max():>10.6f}")
    long_scores = df_signals[df_signals["agreed_long"]]["final_ranking_score"]
    short_scores = df_signals[df_signals["agreed_short"]]["final_ranking_score"]
    print("\n   Score Distribution by Direction:")
    print(
        f"     LONG  (agreed):  mean = {long_scores.mean():>8.4f}, std = {long_scores.std():>8.4f}, n = {len(long_scores):>5,}"
    )
    print(
        f"     SHORT (agreed):  mean = {short_scores.mean():>8.4f}, std = {short_scores.std():>8.4f}, n = {len(short_scores):>5,}"
    )

# Step 8: Comprehensive Method Comparison (summary stats)
mask_agreed = df_signals["agreed_any"]
comparison_stats = []
for method in ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
    scores = df_signals.loc[mask_agreed, f"score_{method}"]
    if len(scores) > 0:
        comparison_stats.append(
            {
                "Method": method,
                "Mean": scores.mean(),
                "Std": scores.std(),
                "Min": scores.min(),
                "Max": scores.max(),
                "Skew": scores.skew(),
                "Count": len(scores),
            }
        )
if comparison_stats:
    print("\nScore Statistics by Method (agreed observations):")
    print(f"   {'Method':<8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Skew':>8}")
    print(f"   {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for stat in comparison_stats:
        print(
            f"   {stat['Method']:<8} {stat['Mean']:>12.6f} {stat['Std']:>12.6f} "
            f"{stat['Min']:>12.6f} {stat['Max']:>12.6f} {stat['Skew']:>8.2f}"
        )
    # Pairwise correlations
    print("\nPairwise Correlations (agreed observations):")
    print(f"   {'':>8} {'S1':>8} {'S2':>8} {'S3':>8} {'S4':>8} {'S5':>8} {'S6':>8} {'S7':>8}")
    print(f"   {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    methods_list = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    for i, method_i in enumerate(methods_list):
        row = f"   {method_i:>8}"
        for j, method_j in enumerate(methods_list):
            if j <= i:
                scores_i = df_signals.loc[mask_agreed, f"score_{method_i}"]
                scores_j = df_signals.loc[mask_agreed, f"score_{method_j}"]
                corr = scores_i.corr(scores_j)
                row += f" {corr:>8.3f}"
            else:
                row += f" {'':>8}"
        print(row)

# Step 9: Store all method results (if ALL mode)
if SCORING_METHOD == "ALL":
    for method in methods_to_process:
        df_signals[f"final_score_{method}"] = all_method_scores[method]

# =============================================================
# 6.2 Allocation Strategies
# =============================================================

print("ALLOCATION STRATEGIES: Portfolio Weight Construction")

# Allocation configuration
ALLOCATION_STRATEGY = "A2"  # Options: "A1"-"A9", or "ALL"

ALLOCATION_DESCRIPTIONS = {
    "A1": "Equal-Weighted: 1/N for each selected stock",
    "A2": "Rank-Weighted: w ∝ score (signal strength)",
    "A3": "Top/Bottom Quantile: Long top x%, Short bottom x%",
    "A4": "Long-Only Threshold: Only long above cutoff",
    "A5": "Volatility-Scaled: w ∝ score/σ (inverse-vol)",
    "A6": "Maximum Sharpe: Mean-variance optimization",
    "A7": "Risk Parity: Equal risk contribution",
    "A8": "Softmax: w ∝ exp(λ·score)",
    "A9": "Kelly Criterion: w ∝ μ/σ² (fractional)",
}

# General Constraints
MAX_POSITION_SIZE = 0.10
LONG_TARGET = 0.5
SHORT_TARGET = 0.5

# Strategy-Specific Parameters
QUANTILE_LONG_PCT = 0.20
QUANTILE_SHORT_PCT = 0.20
LONG_ONLY_THRESHOLD_PERCENTILE = 0.60
SOFTMAX_LAMBDA = 2.0
KELLY_FRACTION = 0.25

# Prepare Data for Allocation
print("\nPreparing Data for Allocation")

# Active scoring method's final ranking score
active_scores = df_signals["final_ranking_score"].copy()

# Agreement masks
long_mask = df_signals["agreed_long"]
short_mask = df_signals["agreed_short"]
agreed_mask = df_signals["agreed_any"]

# Estimate volatility if needed (already computed earlier, but ensure present)
if "volatility" not in df_signals.columns:
    df_signals["volatility"] = df_signals.groupby(level="permno")["adj_prc_logret"].transform(
        lambda x: x.rolling(window=20, min_periods=5).std()
    )
    df_signals["volatility"] = df_signals.groupby(level="date")["volatility"].transform(
        lambda x: x.fillna(x.median())
    )

print("\n  Data prepared:")
print(f"   Total observations: {len(df_signals):,}")
print(f"   Agreed (tradeable): {agreed_mask.sum():,}")
print(f"   Long candidates:    {long_mask.sum():,}")
print(f"   Short candidates:   {short_mask.sum():,}")

# Apply Allocation Strategy (or ALL)
print("\nApplying Allocation Strategy")

returns_df = df_signals[["adj_prc_logret"]].copy()

if ALLOCATION_STRATEGY == "ALL":
    strategies_to_process = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
else:
    strategies_to_process = [ALLOCATION_STRATEGY]

all_strategy_weights = {}

for strategy in strategies_to_process:
    # Prepare parameters
    strategy_params = {
        "long_target": LONG_TARGET,
        "short_target": SHORT_TARGET,
        "max_position_size": MAX_POSITION_SIZE,
    }
    if strategy == "A3":
        strategy_params["quantile_long_pct"] = QUANTILE_LONG_PCT
        strategy_params["quantile_short_pct"] = QUANTILE_SHORT_PCT
    elif strategy == "A4":
        strategy_params["threshold_percentile"] = LONG_ONLY_THRESHOLD_PERCENTILE
    elif strategy == "A8":
        strategy_params["lambda_param"] = SOFTMAX_LAMBDA
    elif strategy == "A9":
        strategy_params["kelly_fraction"] = KELLY_FRACTION

    weights = apply_allocation_strategy(
        strategy_name=strategy,
        scores=active_scores,
        long_mask=long_mask,
        short_mask=short_mask,
        volatility=df_signals["volatility"],
        expected_returns=df_signals["expected_return"],
        returns_df=returns_df,
        **strategy_params,
    )
    all_strategy_weights[strategy] = weights

# Set the active weights
if ALLOCATION_STRATEGY == "ALL":
    df_signals["portfolio_weights"] = all_strategy_weights["A2"]  # default downstream
else:
    df_signals["portfolio_weights"] = all_strategy_weights[ALLOCATION_STRATEGY]

# Final portfolio summary
print("\n" + "=" * 80)
print("Final Portfolio Summary")
print("=" * 80)

active_weights = df_signals["portfolio_weights"]
non_zero_weights = active_weights[active_weights != 0]
n_positions = len(non_zero_weights)
n_long_positions = (active_weights > 0).sum()
n_short_positions = (active_weights < 0).sum()
long_exposure = active_weights[active_weights > 0].sum()
short_exposure = -active_weights[active_weights < 0].sum()
net_exposure = active_weights.sum()
gross_exposure = active_weights.abs().sum()

print(f"\n  Total Positions:       {n_positions:>6,}")
print(f"    Long:                {n_long_positions:>6,}")
print(f"    Short:               {n_short_positions:>6,}")
print("\n  Exposure:")
print(f"    Long:                {long_exposure:>8.2%}")
print(f"    Short:               {short_exposure:>8.2%}")
print(f"    Net:                 {net_exposure:>8.2%}")
print(f"    Gross:               {gross_exposure:>8.2%}")
print("\n  Position Sizing:")
print(f"    Max weight:          {active_weights.abs().max():>8.2%}")
print(f"    Mean |weight|:       {non_zero_weights.abs().mean():>8.2%}")
print(f"    Median |weight|:     {non_zero_weights.abs().median():>8.2%}")

# =============================================================
# 6.3 Optimal Scoring Method/Allocation Strategy Combination
# =============================================================

# Optimization Configuration (prints converted to comments)
TEST_SCORING_METHODS = "ALL"  # "ALL" or list like ["S1","S2","S3"]
TEST_ALLOCATION_STRATEGIES = "ALL"  # "ALL" or list like ["A1","A2","A3"]
OPTIMIZATION_METRIC = "sharpe"  # "sharpe", "total_return", "sortino", "calmar"
MIN_TRADES_REQUIRED = 10
USE_DATA = "in_sample"  # "in_sample" or "full"

# Prepare Test Data
print("\nPreparing Test Data for Optimization")
if USE_DATA == "in_sample":
    test_df = df_signals[df_signals.index.get_level_values("date").isin(ins_dates)].copy()
else:
    test_df = df_signals.copy()

print(f"   Total observations: {len(test_df):,}")
print(
    f"   Date range: {test_df.index.get_level_values('date').min()} to {test_df.index.get_level_values('date').max()}"
)
print(f"   Trading days: {test_df.index.get_level_values('date').nunique()}")

# Define Methods to Test
if TEST_SCORING_METHODS == "ALL":
    scoring_methods_to_test = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
else:
    scoring_methods_to_test = TEST_SCORING_METHODS

if TEST_ALLOCATION_STRATEGIES == "ALL":
    allocation_strategies_to_test = [
        "A1",
        "A2",
        "A3",
        "A5",
        "A7",
        "A8",
    ]  # A4/A6/A9 omitted for speed
else:
    allocation_strategies_to_test = TEST_ALLOCATION_STRATEGIES

score_column_map = {
    "S1": "score_S1",
    "S2": "score_S2",
    "S3": "score_S3",
    "S4": "score_S4",
    "S5": "score_S5",
    "S6": "score_S6",
    "S7": "score_S7",
}

print(f"\n Scoring Methods ({len(scoring_methods_to_test)}):")
for method in scoring_methods_to_test:
    print(f"   - {method}: {SCORING_DESCRIPTIONS.get(method, 'N/A')}")
print(f"\n Allocation Strategies ({len(allocation_strategies_to_test)}):")
for strategy in allocation_strategies_to_test:
    print(f"   - {strategy}: {ALLOCATION_DESCRIPTIONS.get(strategy, 'N/A')}")

total_combinations = len(scoring_methods_to_test) * len(allocation_strategies_to_test)
print(f"\n Total combinations to test: {total_combinations}")

# Run Backtests for All Combinations
print("\nRunning Backtests")
results = []
start_time = time.time()

allocation_params = {
    "long_target": LONG_TARGET,
    "short_target": SHORT_TARGET,
    "max_position_size": MAX_POSITION_SIZE,
    "quantile_long_pct": QUANTILE_LONG_PCT,
    "quantile_short_pct": QUANTILE_SHORT_PCT,
    "threshold_percentile": LONG_ONLY_THRESHOLD_PERCENTILE,
    "lambda_param": SOFTMAX_LAMBDA,
    "kelly_fraction": KELLY_FRACTION,
}

combination_num = 0
for scoring_method in scoring_methods_to_test:
    for allocation_strategy in allocation_strategies_to_test:
        combination_num += 1
        try:
            result = backtest_strategy(
                df=test_df,
                scoring_method=scoring_method,
                allocation_strategy=allocation_strategy,
                score_columns=score_column_map,
                allocation_func=apply_allocation_strategy,
                allocation_params=allocation_params,
                returns_col="adj_prc_logret_lead1",
                date_col="date",
                stock_col="permno",
                rf_rate=0.0,
            )
            n_trades = (result["weights"] != 0).sum()  # noqa
            result_summary = {
                "scoring": scoring_method,
                "allocation": allocation_strategy,
                "n_trades": n_trades,
                **result["metrics"],
            }
            results.append(result_summary)
        except Exception as e:
            results.append(
                {
                    "scoring": scoring_method,
                    "allocation": allocation_strategy,
                    "n_trades": 0,
                    "total_return": np.nan,
                    "sharpe": np.nan,
                    "max_drawdown": np.nan,
                    "error": str(e)[:100],
                }
            )

total_time = time.time() - start_time
print(f"\n  Backtesting complete! Total time: {total_time:.2f}s")
print(f"   Average time per combination: {total_time/total_combinations:.2f}s")

# Analyze Results
print("\nAnalyzing Results")
results_df = pd.DataFrame(results)
valid_results = results_df[results_df["n_trades"] >= MIN_TRADES_REQUIRED].copy()

print(f"\n   Total combinations tested: {len(results_df)}")
print(f"   Valid combinations (≥{MIN_TRADES_REQUIRED} trades): {len(valid_results)}")
print(f"   Invalid/Failed combinations: {len(results_df) - len(valid_results)}")

if len(valid_results) == 0:
    print("\n  No valid combinations found! Adjust MIN_TRADES_REQUIRED or check data.")
else:
    print(f"\n  Ranking by {OPTIMIZATION_METRIC.upper()}")
    valid_results = valid_results.sort_values(by=OPTIMIZATION_METRIC, ascending=False)
    print("\n  TOP 10 COMBINATIONS:\n")
    print(
        f"{'Rank':<6} {'Scoring':<10} {'Allocation':<12} {OPTIMIZATION_METRIC.upper():<10} {'Sharpe':<10} {'Return %':<12} {'MaxDD %':<10} {'Trades':<8}"
    )
    print(f"{'-'*6} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")
    for i, row in valid_results.head(10).iterrows():
        print(
            f"{i+1:<6} {row['scoring']:<10} {row['allocation']:<12} "
            f"{row[OPTIMIZATION_METRIC]:>10.4f} {row['sharpe']:>10.4f} "
            f"{row['total_return']:>12.2f} {row['max_drawdown']:>10.2f} {row['n_trades']:>8,.0f}"
        )

    best = valid_results.iloc[0]
    print("\n  OPTIMAL COMBINATION")
    print(f"\n  Best Scoring Method:      {best['scoring']}")
    print(f"  Best Allocation Strategy: {best['allocation']}")
    print("\n  Performance Metrics:")
    print(f"   {OPTIMIZATION_METRIC.capitalize()}: {best[OPTIMIZATION_METRIC]:.4f}")
    print(f"   Total Return:        {best['total_return']:.2f}%")
    print(f"   Annualized Return:   {best['ann_return']:.2f}%")
    print(f"   Sharpe Ratio:        {best['sharpe']:.4f}")
    print(f"   Sortino Ratio:       {best['sortino']:.4f}")
    print(f"   Calmar Ratio:        {best['calmar']:.4f}")
    print(f"   Max Drawdown:        {best['max_drawdown']:.2f}%")
    print(f"   Win Rate:            {best['win_rate']:.2f}%")
    print(f"   Number of Trades:    {best['n_trades']:,.0f}")

    # Comparison by Scoring Method
    print("\n  Performance by Scoring Method (Best Allocation for Each)")
    scoring_best = (
        valid_results.groupby("scoring")
        .first()
        .sort_values(by=OPTIMIZATION_METRIC, ascending=False)
    )
    print(
        f"\n{'Method':<10} {'Best Alloc':<12} {OPTIMIZATION_METRIC.upper():<12} {'Sharpe':<10} {'Return %':<12}"
    )
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")
    for method, row in scoring_best.iterrows():
        print(
            f"{method:<10} {row['allocation']:<12} {row[OPTIMIZATION_METRIC]:>12.4f} "
            f"{row['sharpe']:>10.4f} {row['total_return']:>12.2f}"
        )

    # Comparison by Allocation Strategy
    print("\n  Performance by Allocation Strategy (Best Scoring for Each)")
    allocation_best = (
        valid_results.groupby("allocation")
        .first()
        .sort_values(by=OPTIMIZATION_METRIC, ascending=False)
    )
    print(
        f"\n{'Strategy':<12} {'Best Score':<12} {OPTIMIZATION_METRIC.upper():<12} {'Sharpe':<10} {'Return %':<12}"
    )
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")
    for strategy, row in allocation_best.iterrows():
        print(
            f"{strategy:<12} {row['scoring']:<12} {row[OPTIMIZATION_METRIC]:>12.4f} "
            f"{row['sharpe']:>10.4f} {row['total_return']:>12.2f}"
        )

    # Store best combination for use in OOS evaluation
    best_scoring = best["scoring"]
    best_allocation = best["allocation"]

# =============================================================
# 7. Out-of-sample evaluation
# =============================================================

print("=" * 80)
print("OUT-OF-SAMPLE EVALUATION")
print("=" * 80)

# 7.1 Prepare OOS Data
# Step: Preparing Out-of-Sample Data

# Define OOS mask
oos_mask = df.index.get_level_values("date").isin(dates_out_sample)

print("OOS Period:")
print(f"  Start Date: {dates_out_sample.min().date()}")
print(f"  End Date:   {dates_out_sample.max().date()}")
print(f"  Trading Days: {len(dates_out_sample)}")
print(f"  Total Observations: {oos_mask.sum():,}")

# 7.2 Generate OOS Predictions
print("\n" + "=" * 80)
print("Step 2: Generating OOS Predictions")
print("=" * 80)

# Logistic Regression Predictions
print("\nLogistic Regression (Direction):")
X_test_logistic = df.loc[oos_mask, final_feature_list]
y_test = DIR_binary.loc[oos_mask]

test_prob_up = pipe_final.predict_proba(X_test_logistic)[:, 1]
test_prob_down = 1 - test_prob_up
test_score = 2 * test_prob_up - 1  # Convert to [-1, +1]
test_pred = pipe_final.predict(X_test_logistic)

print(f"   Predictions: {len(test_prob_up):,}")
print(f"   prob(up) range: [{test_prob_up.min():.4f}, {test_prob_up.max():.4f}]")
print(f"   Mean prob(up): {test_prob_up.mean():.4f}")

# Classification metrics
test_accuracy = (test_pred == y_test).mean()  # noqa
print(f"\n   Classification Accuracy: {test_accuracy:.1%}")
print(f"   Up class accuracy: {(test_pred[y_test == 1] == 1).mean():.1%}")  # noqa
print(f"   Down class accuracy: {(test_pred[y_test == 0] == 0).mean():.1%}")  # noqa

# Linear Regression Predictions
print("\nLinear Regression (Expected Return):")
X_test_linear = df.loc[oos_mask, final_feature_list_linear]
test_expected_return = pipe_final_linear.predict(X_test_linear)

print(f"   Predictions: {len(test_expected_return):,}")
print(f"   E[R] range: [{test_expected_return.min():.6f}, {test_expected_return.max():.6f}]")
print(f"   Mean E[R]: {test_expected_return.mean():.6f}")
print(f"   Std E[R]: {test_expected_return.std():.6f}")

# 7.3 Build OOS DataFrame with All Signals
print("\n" + "=" * 80)
print("Step 3: Building OOS Signal DataFrame")
print("=" * 80)

# Create OOS dataframe with all predictions
oos_df = df.loc[oos_mask].copy()

# Add logistic predictions
oos_df["prob_up"] = pd.Series(test_prob_up, index=X_test_logistic.index)
oos_df["prob_down"] = pd.Series(test_prob_down, index=X_test_logistic.index)
oos_df["logistic_score"] = pd.Series(test_score, index=X_test_logistic.index)
oos_df["pred_class"] = pd.Series(test_pred, index=X_test_logistic.index)

# Add linear predictions
oos_df["expected_return"] = pd.Series(test_expected_return, index=X_test_linear.index)

# 7.4 Apply Ensemble Agreement Filter
print("\nEnsemble Agreement Filter:")

# Determine agreement
oos_df["logistic_signal_long"] = oos_df["prob_up"] > 0.5
oos_df["logistic_signal_short"] = oos_df["prob_up"] < 0.5
oos_df["linear_signal_long"] = oos_df["expected_return"] > 0
oos_df["linear_signal_short"] = oos_df["expected_return"] < 0

# Agreement masks
oos_df["agreed_long"] = oos_df["logistic_signal_long"] & oos_df["linear_signal_long"]
oos_df["agreed_short"] = oos_df["logistic_signal_short"] & oos_df["linear_signal_short"]
oos_df["agreed_any"] = oos_df["agreed_long"] | oos_df["agreed_short"]
oos_df["disagreed"] = ~oos_df["agreed_any"]

print(
    f"   Both agree LONG:  {oos_df['agreed_long'].sum():>6,} ({oos_df['agreed_long'].mean()*100:>5.1f}%)"
)
print(
    f"   Both agree SHORT: {oos_df['agreed_short'].sum():>6,} ({oos_df['agreed_short'].mean()*100:>5.1f}%)"
)
print(
    f"   Total Agreement:  {oos_df['agreed_any'].sum():>6,} ({oos_df['agreed_any'].mean()*100:>5.1f}%)"
)
print(
    f"   Disagreement:     {oos_df['disagreed'].sum():>6,} ({oos_df['disagreed'].mean()*100:>5.1f}%)"
)

# 7.5 Calculate All Scoring Methods on OOS Data
print("\n" + "=" * 80)
print("Step 4: Calculating All Scoring Methods on OOS")
print("=" * 80)

# Use processed variables
p = oos_df["prob_up"]
mu = oos_df["expected_return"]

# Apply same preprocessing as in Section 6.1
# Winsorization
if USE_WINSORIZATION:

    def winsorize_cross_sectional(series, limits):
        lower, upper = limits
        q_lower = series.quantile(lower)
        q_upper = series.quantile(upper)
        return series.clip(lower=q_lower, upper=q_upper)

    p_processed = oos_df.groupby(level="date")["prob_up"].transform(
        lambda x: winsorize_cross_sectional(x, WINSORIZE_LIMITS)
    )
    mu_processed = oos_df.groupby(level="date")["expected_return"].transform(
        lambda x: winsorize_cross_sectional(x, WINSORIZE_LIMITS)
    )
else:
    p_processed = p
    mu_processed = mu

# Cross-sectional standardization
if USE_CROSS_SECTIONAL_STANDARDIZATION:

    def z_score_cross_sectional(series):
        mean = series.mean()
        std = series.std()
        if std > 0:
            return (series - mean) / std
        return series - mean

    p_tilde = oos_df.groupby(level="date")["prob_up"].transform(z_score_cross_sectional)
    mu_tilde = oos_df.groupby(level="date")["expected_return"].transform(z_score_cross_sectional)
else:
    p_tilde = p_processed
    mu_tilde = mu_processed

# Calculate S1-S7
print("\n   Calculating S1-S7...")

oos_df["score_S1"] = p_processed * mu_processed
oos_df["score_S2"] = (p_processed - 0.5) * mu_processed
p_clipped = p_processed.clip(lower=0.01, upper=0.99)
logit_p = np.log(p_clipped / (1 - p_clipped))
oos_df["score_S3"] = logit_p * mu_processed
oos_df["score_S4"] = ALPHA_S4 * p_tilde + (1 - ALPHA_S4) * mu_tilde
oos_df["score_S5"] = p_tilde * mu_tilde
oos_df["score_S6"] = (2 * p_processed - 1) * mu_processed.abs()

if "volatility" not in oos_df.columns:
    oos_df["volatility"] = oos_df.groupby(level="permno")["adj_prc_logret"].transform(
        lambda x: x.rolling(window=20, min_periods=5).std()
    )
    oos_df["volatility"] = oos_df.groupby(level="date")["volatility"].transform(
        lambda x: x.fillna(x.median())
    )
oos_df["score_S7"] = (p_processed * mu_processed) / (oos_df["volatility"] + 1e-8)

print("   All scoring methods calculated")

# Ensemble filter: set scores to 0 where models disagree
for score_col in [
    "score_S1",
    "score_S2",
    "score_S3",
    "score_S4",
    "score_S5",
    "score_S6",
    "score_S7",
]:
    oos_df.loc[oos_df["disagreed"], score_col] = 0.0

print("   Ensemble filter applied (scores set to 0 for disagreements)")

print("\nOOS DataFrame prepared:")
print(f"   Shape: {oos_df.shape}")
print(f"   New columns added: {len([c for c in oos_df.columns if 'score_' in c or 'agreed' in c])}")

# 7.6 Apply Optimal Strategy (from Section 6.3)

print("\nApplying Optimal Strategy from Section 6.3\n")

if "best_scoring" not in locals() or "best_allocation" not in locals():
    print("WARNING: Optimization results not found! Using default: S2 + A2")
    best_scoring = "S2"
    best_allocation = "A2"

print("Optimal Combination:")
print(f"  Scoring Method:      {best_scoring} - {SCORING_DESCRIPTIONS.get(best_scoring, 'N/A')}")
print(
    f"  Allocation Strategy: {best_allocation} - {ALLOCATION_DESCRIPTIONS.get(best_allocation, 'N/A')}"
)


# Helper to apply strategy on OOS and compute metrics
def apply_strategy_oos(df_oos, scoring_method, allocation_strategy, label="Strategy"):
    print(f"\n{'-'*80}")
    print(f"Evaluating: {label}")
    print(f"{'-'*80}")

    score_col = f"score_{scoring_method}"
    if score_col not in df_oos.columns:
        raise ValueError(f"Score column {score_col} not found")

    weights_list = []
    dates = df_oos.index.get_level_values("date").unique()

    for date in dates:
        date_df = df_oos.loc[df_oos.index.get_level_values("date") == date]
        scores = date_df[score_col]
        long_mask = date_df["agreed_long"]
        short_mask = date_df["agreed_short"]
        volatility = date_df["volatility"]
        expected_returns = date_df["expected_return"]

        alloc_params = {
            "long_target": LONG_TARGET,
            "short_target": SHORT_TARGET,
            "max_position_size": MAX_POSITION_SIZE,
        }
        if allocation_strategy == "A3":
            alloc_params["quantile_long_pct"] = QUANTILE_LONG_PCT
            alloc_params["quantile_short_pct"] = QUANTILE_SHORT_PCT
        elif allocation_strategy == "A4":
            alloc_params["threshold_percentile"] = LONG_ONLY_THRESHOLD_PERCENTILE
        elif allocation_strategy == "A8":
            alloc_params["lambda_param"] = SOFTMAX_LAMBDA
        elif allocation_strategy == "A9":
            alloc_params["kelly_fraction"] = KELLY_FRACTION

        try:
            weights = apply_allocation_strategy(
                strategy_name=allocation_strategy,
                scores=scores,
                long_mask=long_mask,
                short_mask=short_mask,
                volatility=volatility,
                expected_returns=expected_returns,
                **alloc_params,
            )
            weights_list.append(weights)
        except Exception as e:
            print(f"   Warning: Allocation failed for {date}: {str(e)[:50]}")
            weights_list.append(pd.Series(0.0, index=date_df.index))

    df_oos_copy = df_oos.copy()
    df_oos_copy["portfolio_weights"] = pd.concat(weights_list)

    portfolio_returns = calculate_portfolio_returns(
        df_oos_copy,
        weights_col="portfolio_weights",
        returns_col="adj_prc_logret_lead1",
        date_col="date",
    )
    metrics = calculate_performance_metrics(portfolio_returns, rf_rate=0.0)
    equity = calculate_equity_curve(portfolio_returns)

    n_trades = (df_oos_copy["portfolio_weights"] != 0).sum()  # noqa
    n_long = (df_oos_copy["portfolio_weights"] > 0).sum()
    n_short = (df_oos_copy["portfolio_weights"] < 0).sum()

    print("\nPerformance Metrics:")
    print(f"   Total Return:      {metrics['total_return']:>10.2f}%")
    print(f"   Annualized Return: {metrics['ann_return']:>10.2f}%")
    print(f"   Annualized Vol:    {metrics['ann_vol']:>10.2f}%")
    print(f"   Sharpe Ratio:      {metrics['sharpe']:>10.4f}")
    print(f"   Sortino Ratio:     {metrics['sortino']:>10.4f}")
    print(f"   Calmar Ratio:      {metrics['calmar']:>10.4f}")
    print(f"   Max Drawdown:      {metrics['max_drawdown']:>10.2f}%")
    print(f"   Win Rate:          {metrics['win_rate']:>10.2f}%")

    print("\nTrading Statistics:")
    print(f"   Total Trades:      {n_trades:>10,}")
    print(f"   Long Positions:    {n_long:>10,}")
    print(f"   Short Positions:   {n_short:>10,}")

    return {
        "label": label,
        "scoring": scoring_method,
        "allocation": allocation_strategy,
        "metrics": metrics,
        "returns": portfolio_returns,
        "equity": equity,
        "weights": df_oos_copy["portfolio_weights"],
        "n_trades": n_trades,
    }


# 7.7 Evaluate Optimal Strategy
print("\n" + "=" * 80)
print("OPTIMAL STRATEGY EVALUATION")
print("=" * 80)

result_optimal = apply_strategy_oos(
    oos_df,
    scoring_method=best_scoring,
    allocation_strategy=best_allocation,
    label=f"Optimal ({best_scoring} + {best_allocation})",
)

# 7.8 Evaluate Benchmarks
print("\n" + "=" * 80)
print("BENCHMARK COMPARISONS")
print("=" * 80)

# Benchmark 1: Naive Baseline (S1 + A1)
print("\nBenchmark 1: Naive Baseline")
result_baseline = apply_strategy_oos(
    oos_df, scoring_method="S1", allocation_strategy="A1", label="Naive Baseline (S1 + A1)"
)

# Benchmark 2: Simple Rank-Weighted (S2 + A2)
if best_scoring != "S2" or best_allocation != "A2":
    print("\nBenchmark 2: Standard Approach")
    result_standard = apply_strategy_oos(
        oos_df, scoring_method="S2", allocation_strategy="A2", label="Standard (S2 + A2)"
    )
else:
    result_standard = None
    print("\nBenchmark 2: Skipped (same as optimal)")

# 7.9 Comparison Table
print("\n" + "=" * 80)
print("COMPREHENSIVE COMPARISON")
print("=" * 80)

comparison_data = []
comparison_data.append(
    {
        "Strategy": result_optimal["label"],
        "Total Return %": result_optimal["metrics"]["total_return"],
        "Ann. Return %": result_optimal["metrics"]["ann_return"],
        "Ann. Vol %": result_optimal["metrics"]["ann_vol"],
        "Sharpe": result_optimal["metrics"]["sharpe"],
        "Sortino": result_optimal["metrics"]["sortino"],
        "Max DD %": result_optimal["metrics"]["max_drawdown"],
        "Win Rate %": result_optimal["metrics"]["win_rate"],
        "Trades": result_optimal["n_trades"],
    }
)
comparison_data.append(
    {
        "Strategy": result_baseline["label"],
        "Total Return %": result_baseline["metrics"]["total_return"],
        "Ann. Return %": result_baseline["metrics"]["ann_return"],
        "Ann. Vol %": result_baseline["metrics"]["ann_vol"],
        "Sharpe": result_baseline["metrics"]["sharpe"],
        "Sortino": result_baseline["metrics"]["sortino"],
        "Max DD %": result_baseline["metrics"]["max_drawdown"],
        "Win Rate %": result_baseline["metrics"]["win_rate"],
        "Trades": result_baseline["n_trades"],
    }
)
if result_standard is not None:
    comparison_data.append(
        {
            "Strategy": result_standard["label"],
            "Total Return %": result_standard["metrics"]["total_return"],
            "Ann. Return %": result_standard["metrics"]["ann_return"],
            "Ann. Vol %": result_standard["metrics"]["ann_vol"],
            "Sharpe": result_standard["metrics"]["sharpe"],
            "Sortino": result_standard["metrics"]["sortino"],
            "Max DD %": result_standard["metrics"]["max_drawdown"],
            "Win Rate %": result_standard["metrics"]["win_rate"],
            "Trades": result_standard["n_trades"],
        }
    )

comparison_df = pd.DataFrame(comparison_data)

print("\nPerformance Comparison Table:\n")
print(comparison_df.to_string(index=False))

# Improvement vs baseline
baseline_sharpe = result_baseline["metrics"]["sharpe"]
optimal_sharpe = result_optimal["metrics"]["sharpe"]
sharpe_improvement = optimal_sharpe - baseline_sharpe
sharpe_improvement_pct = (
    (sharpe_improvement / abs(baseline_sharpe) * 100) if baseline_sharpe != 0 else 0
)

baseline_return = result_baseline["metrics"]["total_return"]
optimal_return = result_optimal["metrics"]["total_return"]
return_improvement = optimal_return - baseline_return

print("\nOptimal vs Baseline:")
print(f"   Sharpe Improvement:  {sharpe_improvement:>+8.4f} ({sharpe_improvement_pct:>+6.1f}%)")
print(f"   Return Improvement:  {return_improvement:>+8.2f}%")

# Store results for Section 8
oos_result_optimal = result_optimal
oos_result_baseline = result_baseline
if result_standard is not None:
    oos_result_standard = result_standard

print("\nOOS Evaluation Complete!")
print("   Results stored in: oos_result_optimal, oos_result_baseline")

# =============================================================
# 8. Output Generation
# =============================================================

print("OUTPUT GENERATION: QuantStats HTML Reports")

# Configuration
GENERATE_OPTIMAL_REPORT = True
GENERATE_BASELINE_REPORT = False
GENERATE_STANDARD_REPORT = False
GENERATE_MONTHLY_REPORT = True
OUTPUT_DIR = "outputs"

# Check Required Variables
print("\nChecking Required Variables")
required_vars = {
    "oos_result_optimal": "oos_result_optimal",
    "oos_df": "oos_df",
    "dates_out_sample": "dates_out_sample",
    "best_scoring": "best_scoring",
    "best_allocation": "best_allocation",
}
missing_vars = []
for var_name, display_name in required_vars.items():
    if var_name not in locals() and var_name not in globals():
        missing_vars.append(display_name)
        print(f"   MISSING: {display_name}")
    else:
        print(f"   Found: {display_name}")

if missing_vars:
    print(f"\nWARNING: Missing required variables: {', '.join(missing_vars)}")
    raise ValueError("Missing required variables from Section 7")

print("\nAll required variables found!")

# Generate Report for Optimal Strategy (Daily)
if GENERATE_OPTIMAL_REPORT:
    print("\n" + "=" * 80)
    print("Generating Report: Optimal Strategy (Daily)")
    print("=" * 80)

    print("\nStrategy Details:")
    print(f"  Scoring Method:      {best_scoring}")
    print(f"  Allocation Strategy: {best_allocation}")
    print("  Performance:")
    print(f"    Total Return:    {oos_result_optimal['metrics']['total_return']:>10.2f}%")
    print(f"    Sharpe Ratio:    {oos_result_optimal['metrics']['sharpe']:>10.4f}")
    print(f"    Max Drawdown:    {oos_result_optimal['metrics']['max_drawdown']:>10.2f}%")

    output_path_optimal = os.path.join(OUTPUT_DIR, "oos_optimal_tearsheet.html")
    report_title_optimal = (
        f"OUT-OF-SAMPLE: Optimal Strategy ({best_scoring} + {best_allocation})\n"
        f"Period: {dates_out_sample.min().date()} to {dates_out_sample.max().date()}"
    )

    generate_oos_report(
        portfolio_result=oos_result_optimal,
        oos_df=oos_df,
        dates_out_sample=dates_out_sample,
        output_path=output_path_optimal,
        report_title=report_title_optimal,
        output_dir=OUTPUT_DIR,
    )

    print("\nOptimal strategy report (daily) generated.")

    # Monthly-aggregated report for the same strategy
    if GENERATE_MONTHLY_REPORT:
        output_path_optimal_m = os.path.join(OUTPUT_DIR, "oos_optimal_tearsheet_monthly.html")
        report_title_optimal_m = (
            f"OUT-OF-SAMPLE (Monthly): Optimal Strategy ({best_scoring} + {best_allocation})\n"
            f"Period: {dates_out_sample.min().date()} to {dates_out_sample.max().date()}"
        )
        generate_oos_report_monthly(
            portfolio_result=oos_result_optimal,
            oos_df=oos_df,
            dates_out_sample=dates_out_sample,
            output_path=output_path_optimal_m,
            report_title=report_title_optimal_m,
            output_dir=OUTPUT_DIR,
        )
        print("Optimal strategy report (monthly) generated.")

# Baseline Report (only daily)
if GENERATE_BASELINE_REPORT:
    print("\n" + "=" * 80)
    print("Generating Report: Baseline Strategy (Daily)")
    print("=" * 80)

    if "oos_result_baseline" not in locals() and "oos_result_baseline" not in globals():
        print("   oos_result_baseline not found. Skipping baseline report.")
    else:
        output_path_baseline = os.path.join(OUTPUT_DIR, "oos_baseline_tearsheet.html")
        report_title_baseline = (
            f"OUT-OF-SAMPLE: Baseline Strategy (S1 + A1)\n"
            f"Period: {dates_out_sample.min().date()} to {dates_out_sample.max().date()}"
        )
        generate_oos_report(
            portfolio_result=oos_result_baseline,
            oos_df=oos_df,
            dates_out_sample=dates_out_sample,
            output_path=output_path_baseline,
            report_title=report_title_baseline,
            output_dir=OUTPUT_DIR,
        )
        print("\nBaseline report (daily) generated.")

# Standard Approach Report (only daily)
if GENERATE_STANDARD_REPORT:
    print("\n" + "=" * 80)
    print("Generating Report: Standard Approach (Daily)")
    print("=" * 80)

    if "oos_result_standard" not in locals() and "oos_result_standard" not in globals():
        print("   oos_result_standard not found. Skipping standard report.")
    else:
        output_path_standard = os.path.join(OUTPUT_DIR, "oos_standard_tearsheet.html")
        report_title_standard = (
            f"OUT-OF-SAMPLE: Standard Approach (S2 + A2)\n"
            f"Period: {dates_out_sample.min().date()} to {dates_out_sample.max().date()}"
        )
        generate_oos_report(
            portfolio_result=oos_result_standard,
            oos_df=oos_df,
            dates_out_sample=dates_out_sample,
            output_path=output_path_standard,
            report_title=report_title_standard,
            output_dir=OUTPUT_DIR,
        )
        print("\nStandard approach report (daily) generated.")

# Final Summary
print("\nSUMMARY: Output Generation Complete")
print(f"\n  Output Directory: {OUTPUT_DIR}/")
print("\n  Reports Generated:")

reports_generated = []
if GENERATE_OPTIMAL_REPORT:
    reports_generated.append(
        ("oos_optimal_tearsheet.html", "Optimal Strategy (Daily)", best_scoring, best_allocation)
    )
    if GENERATE_MONTHLY_REPORT:
        reports_generated.append(
            (
                "oos_optimal_tearsheet_monthly.html",
                "Optimal Strategy (Monthly)",
                best_scoring,
                best_allocation,
            )
        )
if GENERATE_BASELINE_REPORT and (
    "oos_result_baseline" in locals() or "oos_result_baseline" in globals()
):
    reports_generated.append(("oos_baseline_tearsheet.html", "Baseline (Daily)", "S1", "A1"))
if GENERATE_STANDARD_REPORT and (
    "oos_result_standard" in locals() or "oos_result_standard" in globals()
):
    reports_generated.append(("oos_standard_tearsheet.html", "Standard (Daily)", "S2", "A2"))

for i, (filename, label, scoring, allocation) in enumerate(reports_generated, 1):
    full_path = os.path.join(OUTPUT_DIR, filename)
    file_exists = os.path.exists(full_path)
    status = "[OK]" if file_exists else "[MISSING]"
    print(f"\n{i}. {status} {filename}")
    print(f"   Label: {label}")
    print(f"   Strategy: {scoring} + {allocation}")
    if file_exists:
        file_size = os.path.getsize(full_path) / 1024  # KB
        print(f"   Size: {file_size:.1f} KB")

if not reports_generated:
    print("\n   No reports were generated. Check configuration flags.")
