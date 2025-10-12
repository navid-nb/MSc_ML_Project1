# Data Processing Pipeline

This section explains the key data cleaning, transformation, and feature engineering steps. The pipeline has been modularized into separate helper functions for better maintainability and testing. Each step summarizes what is done and why it is needed.

---

## 1. Data Loading and Filtering

**Functions:** `parquet_to_df()`, `filter_by_tickers()`, `filter_by_tickers_and_permno_pairs()`, `common_features_extract()`

- **Raw extraction from WRDS** loads pre-computed Parquet datasets from configured SQL queries:
  - Daily Stock File (DSF) from `dsf.parquet`
  - Fama-French daily factors from `ff.parquet`
  - IBES summary consensus from `ibes_stats.parquet` 
  - IBES actual EPS announcements from `ibes_act.parquet`

- **Yahoo Finance download** via `common_features_extract()` provides market-wide and cross-asset data:
  - **Volatility indices**: VIX (S&P 500 fear gauge), VXN (NASDAQ volatility), OVX (Oil volatility), GVZ (Gold volatility)
  - **Market indices**: S&P 500 (^GSPC), NASDAQ Composite (^IXIC), Russell 2000 (^RUT)
  - **Sector ETFs**: XLK (Technology), XLF (Financials), XLE (Energy), XLV (Healthcare), XLI (Industrials)
  - Data saved to `common_features.parquet` with standardized column naming (e.g., `comm_^VIX_close`)

- **Ticker filtering** is applied to focus on specified stocks using `filter_by_tickers()`
- **Cross-dataset alignment** ensures IBES and FF data only includes permno-date pairs present in the main DSF dataset using `filter_by_tickers_and_permno_pairs()`

*Purpose:* Load and filter raw datasets to focus on the target universe of stocks while incorporating broader market context for modeling.

---

## 2. Index Setup and Data Quality Assurance

**Functions:** `ensure_index()`, `pre_qa_dsf()`, `clean_dsf()`

- **Set MultiIndex (permno, date)** on DSF using `ensure_index()` for efficient time-series operations
- **Pre-processing quality checks** via `pre_qa_dsf()` include:
  - Date validity and type enforcement
  - Check adjustment factors (cfacpr, cfacshr) for zeros or negatives
  - Report negative or zero prices occurrences
  - Ensure uniqueness of (permno, date) key
- **Data cleaning** with `clean_dsf()` removes problematic records and applies corrections

*Purpose:* Ensure data integrity and establish proper indexing for downstream time-based operations.

---

## 3. Sequential Data Integration

The pipeline now performs sequential joins with quality assurance at each step:

### 3.1 Fama-French Factors Integration
**Functions:** `pre_qa_ff()`, `join_prices_with_ff()`, `post_join_qa_prices_with_ff()`

- Pre-join validation of FF data quality and date alignment
- Join daily stock prices with Fama-French market factors by date
- Post-join validation to ensure data integrity and proper alignment

### 3.2 IBES Consensus Data Integration  
**Functions:** `pre_qa_ibes_statsumu()`, `prepare_ibes_for_daily_merge()`, `join_prices_with_ibes()`, `post_join_qa_prices_with_ibes()`

- Quality checks on IBES consensus summary data
- Transform IBES data to daily granularity for merging
- Join consensus forecasts with price data
- Validate successful integration and coverage

### 3.3 IBES Actuals Integration
**Functions:** `pre_qa_ibes_actu()`, `prepare_ibes_actu_for_daily_merge()`, `join_prices_with_ibes_actu()`, `post_join_qa_prices_with_ibes_actu()`

- Validate IBES actual earnings announcement data
- Prepare actuals for daily merge alignment
- Integrate actual earnings data with the growing dataset
- Quality assurance on the expanded dataset

### 3.4 Yahoo Finance Common Features Integration
**Functions:** `join_prices_with_common_features()`

- Join market-wide and cross-asset indicators from Yahoo Finance data
- These include volatility indices (VIX, VXN, OVX, GVZ), market indices (S&P 500, NASDAQ, Russell 2000), and sector ETFs (XLK, XLF, XLE, XLV, XLI)
- Provides broader market context and regime indicators for enhanced modeling

*Purpose:* Systematically build a comprehensive dataset while maintaining data quality at each integration step.

---

## 4. Missing Value Treatment

**Functions:** `forward_fill_and_remove_initial_nans()`

- Apply forward-fill imputation within each `permno` group to fill temporal gaps in numeric features
- Remove leading rows with NaNs that cannot be forward-filled to maintain clean time series
- Optional tracking of source dates for forward-filled values for transparency
- Report any remaining missing values post-imputation for further attention

*Purpose:* Ensure continuous, gap-free time series data for each stock required by machine learning models.

---

## 5. Feature Augmentation

**Functions:** `feature_augmentation()`, `_add_technical_indicators()`

- **Technical indicators** are computed using pandas-ta library including:
  - Momentum indicators: RSI, MACD, Stochastic Oscillator
  - Volatility measures: ATR (Average True Range), Bollinger Bands
  - Trend indicators: ADX, Aroon, Parabolic SAR
  - Volume analysis: Money Flow Index (MFI), Chaikin Money Flow (CMF), Ease of Movement (EOM)
  - Statistical measures: Rolling variance, skewness, kurtosis
- **Cross-asset ratio features** from VIX, market indices, and sector ETFs. including VIX/S&P500 ratio for normalized fear measurement, sector ETF/S&P500 ratios (XLK, XLF, XLE, XLV, XLI) for rotation detection, volatility premium calculations (VIX minus realized volatility), and market beta proxies for stock sensitivity measurement. These ratios capture market regime changes, sector rotation patterns, volatility risk premiums, and individual stock market sensitivity crucial for systematic trading strategies.
- **Temporal features** including lagged returns and momentum indicators
- **Feature cleaning** removes rows with all NaNs after indicator calculation

*Purpose:* Transform raw price/volume data into predictive features while maintaining temporal alignment.

---

## 6. Final Matrix Construction

**Functions:** `build_final_matrix()`, `align_and_fill_dates_across_tickers()`

- **Target variable creation**: Set next-day log return (`adj_prc_logret_lead1`) as the prediction target
- **Feature selection**: Choose core predictive features including:
  - Price and volume data
  - Technical indicators 
  - Fundamental variables (IBES consensus/actuals)
  - Lagged market factors (FF) to avoid look-ahead bias
  - Yahoo Finance market indicators (VIX, sector ETFs, market indices) for cross-asset signals
- **Cross-stock alignment**: Ensure all stocks have consistent date coverage using `align_and_fill_dates_across_tickers()`
- **Final cleaning**: Remove any remaining rows with missing essential features

*Purpose:* Create a temporally aligned, complete feature matrix ready for machine learning with proper target variable definition.

---

## 7. Data Quality Validation and Output

- **Comprehensive diagnostics** report final matrix shape, index structure, and column composition
- **Target variable validation** with statistical summary of `adj_prc_logret_lead1`:
  - Distribution analysis (mean, median, standard deviation)
  - Extreme value detection (returns < -1.0 or > 1.0)
  - Data range validation for realistic return magnitudes
- **Final structure verification** ensures proper MultiIndex (permno, date) for downstream modeling

*Purpose:* Provide comprehensive quality assurance and transparency for the final modeling dataset.

---

---

# Key Processing Functions by Module

## Data Cleanup (`functions.helpers.data_cleanup`)
- **`parquet_to_df()`**: Load Parquet files from WRDS extracts
- **`filter_by_tickers()`**: Filter datasets to specified stock universe  
- **`ensure_index()`**: Set proper MultiIndex structure
- **`pre_qa_*()` functions**: Pre-processing quality assurance for each dataset
- **`clean_dsf()`**: Clean and validate daily stock file data
- **`join_prices_with_*()` functions**: Sequential data integration with validation
- **`post_join_qa_*()` functions**: Post-integration quality checks

## Feature Engineering (`functions.helpers.feature_engineering`)
- **`feature_augmentation()`**: Main feature engineering orchestration
- **`_add_technical_indicators()`**: Compute technical indicators using pandas-ta
- **`build_final_matrix()`**: Construct final modeling matrix with target variable

## Data Extraction (`functions.helpers.data_extraction`)
- **`wrds_extract_raw()`**: Extract raw data from WRDS using SQL queries

---

# Updated Data Processing Flow

The current pipeline follows this modular flow:

1. **Load & Filter** → Raw Parquet loading and ticker filtering
2. **Index & QA** → MultiIndex setup and initial quality checks  
3. **Sequential Integration** → Step-by-step data joins with validation
4. **Missing Value Treatment** → Forward-fill imputation and cleaning
5. **Feature Engineering** → Technical indicators and cross-asset features
6. **Final Matrix** → Target variable creation and alignment
7. **Quality Validation** → Comprehensive output validation

---

## 9. Final Clean-Up and Output

- Reset the DataFrame index, typically dropping the ‘date’ level index to flatten the structure for modeling.
- Print informative diagnostics about shape, index levels, columns, and missing data status of the final matrix.
- Return the fully prepared modeling DataFrame ready for machine learning workflows.
  
*Purpose:* Provide a clean, well-structured dataset ensuring transparency and reliability for downstream tasks.

---

# Key Data Quality Actions

| Action                            | Function/Module | Purpose / Reason                                                                                           |
|----------------------------------|-----------------|-----------------------------------------------------------------------------------------------------------|
| Ticker-based filtering | `filter_by_tickers()` | Focus analysis on specified stock universe and reduce computational overhead |
| MultiIndex enforcement | `ensure_index()` | Enable efficient time-series operations and groupby functionality |
| Pre-join validation | `pre_qa_*()` functions | Catch data quality issues before integration to prevent propagation |
| Sequential integration | `join_prices_with_*()` | Build comprehensive dataset while maintaining quality at each step |
| Post-join validation | `post_join_qa_*()` | Verify successful integration and detect any anomalies introduced |
| Forward-fill imputation | `forward_fill_and_remove_initial_nans()` | Fill temporal gaps while preserving time-series structure |
| Technical indicator computation | `_add_technical_indicators()` | Transform raw OHLCV data into predictive features |
| Cross-stock alignment | `align_and_fill_dates_across_tickers()` | Ensure consistent date coverage across all stocks |
| Target variable validation | Quality check in main function | Verify realistic return distributions and detect outliers |

---

