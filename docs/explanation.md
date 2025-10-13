# Data Processing Pipeline

This section walks through how data is cleaned, transformed, and
enriched before modeling.

---

## 1. Loading and Filtering Data

**Functions:** `parquet_to_df()`, `filter_by_tickers()`, `filter_by_tickers_and_permno_pairs()`,
`common_features_extract()`

- **Data sources:**
    - **WRDS extracts**: Parquet files for stock prices (DSF), Fama-French factors (FF), and IBES earnings data (
      summary + actuals).
    - **Yahoo Finance**: Market-wide indicators (VIX, sector ETFs, indices like S&P 500 or NASDAQ).

- **Filtering:**  
  We only keep the target tickers and make sure all datasets (DSF, FF, IBES) align on the same date and `permno` keys.

**Goal:** Start with source datasets and include useful market context.

---

## 2. Indexing and Quality Checks

**Functions:** `ensure_index()`, `pre_qa_dsf()`, `clean_dsf()`

- Set a **MultiIndex (permno, date)**.
- Run checks on dates, prices, adjustment factors, and uniqueness of records.
- Clean up or remove problematic rows.

**Goal:** Guarantee data consistency before merging tables together.

---

## 3. Step-by-Step Data Integration

We merge datasets one by one, with checks after every join to avoid spreading errors.

### 3.1 Fama-French Factors

Join daily stock data with market risk factors (FF).

### 3.2 IBES Consensus Forecasts

Add daily forecast data from analysts' earnings expectations (IBES summary).

### 3.3 IBES Actual Announcements

Add actual earnings results.

### 3.4 Yahoo Finance Features

Include macro and sector signals, like volatility indices (VIX, VXN) and ETFs (XLK, XLF, etc.).

**Goal:** Combine all relevant signals into a final matrix for modelling.

---

## 4. Handling Missing Data

**Function:** `forward_fill_and_remove_initial_nans()`

- Forward-fill missing values within each stock to bridge small gaps.
- Drop early rows that can't be filled.

**Goal:** Create gap-free time series matrix for modelling.

---

## 5. Feature Engineering and Enrichment

**Functions:** `feature_augmentation()`, `_add_technical_indicators()`

- Compute **technical indicators** (RSI, MACD, ATR, ADX, Bollinger Bands, etc.) using `pandas-ta`.
- Create **cross-asset ratios** (like VIX/S&P500 or sector ETF ratios) to capture market regimes, rotations, and
  volatility spreads.
- Add **lagged returns and temporal features** for momentum signals.
- Drop rows that remain entirely NaN after feature creation.

**Goal:** Turn raw time-series data into a matrix with predictive features.

---

## 6. Building the Final Dataset

**Functions:** `build_final_matrix()`, `align_and_fill_dates_across_tickers()`

- Define the **target variable**: next-day log return (`adj_prc_logret_lead1`).
- Select the main features (price, volume, technical, fundamental, and macro).
- Align all tickers to ensure consistent date coverage.
- Final cleanup to remove missing essentials.

**Goal:** Produce a feature matrix for model training.

---

## 7. Final Quality Validation

- Check the shape, index, and structure of the final dataset.
- Validate the target variable's statistics: mean, variance, and outliers.

**Goal:** Confirm that the final dataset is ready to use.

---

## Key Functions and Modules

| Category                   | Module                                  | Key Functions                                                                                                                       |
|----------------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **Data Loading & Cleanup** | `functions.helpers.data_cleanup`        | `parquet_to_df()`, `filter_by_tickers()`, `ensure_index()`, `clean_dsf()`, `join_prices_with_*()`, `pre_qa_*()`, `post_join_qa_*()` |
| **Feature Engineering**    | `functions.helpers.feature_engineering` | `feature_augmentation()`, `_add_technical_indicators()`, `build_final_matrix()`                                                     |
| **Data Extraction**        | `functions.helpers.data_extraction`     | `wrds_extract_raw()`                                                                                                                |

---

## Core Quality Controls

| Step                  | Function                                 | Why It Matters                   |
|-----------------------|------------------------------------------|----------------------------------|
| Ticker filtering      | `filter_by_tickers()`                    | Ensures ticker consistency       |
| Indexing              | `ensure_index()`                         | Enables time-series ops          |
| Pre/Post QA           | `pre_qa_*()` / `post_join_qa_*()`        | Prevents data errors             |
| Forward-fill          | `forward_fill_and_remove_initial_nans()` | Maintains continuous time series |
| Technical indicators  | `_add_technical_indicators()`            | Converts raw data into signals   |
| Cross-stock alignment | `align_and_fill_dates_across_tickers()`  | Ensures equal date coverage      |
| Target validation     | Main function QA                         | Detects outliers                 |