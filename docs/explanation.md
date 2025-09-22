# Data Processing Steps in `build_model_matrix_from_wrds`

This section explains the key data cleaning, transformation, and feature engineering steps in the main function `build_model_matrix_from_wrds` for preparing financial modeling data from WRDS extracts. Each step summarizes what is done and why it is needed.

---

## 1. Data Extraction and Loading

- **Raw extraction from WRDS** is run using configured SQL queries producing Parquet datasets for:
  - Daily Stock File (DSF)
  - CRSP stock names
  - Fama-French daily factors (FF)
  - IBES summary consensus (ibes_statsumu)
  - IBES actual EPS announcements (ibes_actu)

- **Loading Parquet files** into individual Pandas DataFrames for processing.

*Purpose:* Initial step to extract necessary raw tables needed for modeling.

---

## 2. Indexing and Preliminary Data Quality Assurance (QA)

- **Set MultiIndex (permno, date)** on DSF for time-series group operations.
- Perform **preliminary quality checks on DSF** such as:
  - Date validity and type enforcement
  - Check adjustment factors (cfacpr, cfacshr) for zeros or negatives (warning for zero/error for negative value).
  - Report negative or zero prices occurrences
  - Ensure uniqueness of (permno, date) key

*Purpose:* Ensure data integrity and correct indexing for downstream time-based operations.

---

## 3. DSF Quality Checks & Cleaning

- **Remove stocks where adjustment factors contain zeros.**  
    Adjustment factors like `cfacpr` and `cfacshr` should be strictly positive; zeros indicate bad or incomplete data which should be excluded.

- **Handle negative prices in DSF:**  
  In WRDS CRSP Daily Stock File, negative price values are not errors and do not indicate actual negative prices. Instead, a negative sign indicates that the reported price is a bid/ask average rather than an official closing price. For example, this can occur on days when no trades actually happened. This means such values are not fundamentally invalid and can contain useful information about price levels on days without closing prices. However, if a stock has an excessive fraction of negative prices exceeding a threshold (1%), it usually indicates poor data quality for that stock, so those stocks are removed to reduce noise in modeling.


- **Convert remaining price values to absolute values**  
  After removing the stocks with poor data quality, the remaning prices are converted to their absolute values so that all price data is consistent and positive for further analysis.

- **Recalculate adjusted prices, shares, and market caps** using cleaned data.


*Purpose:* Ensure dsf data is clean, consistent, and correctly adjusted for splits/dividends.

---

## 4. Merging Stock Names 
4.1- pre join quality checks 
- Verify date types and check for overlapping validity windows in stock naming history.
- Warns if overlapping name windows exist for the same permno.

4.2- join stocknames to dsf

- Performs a left "as-of" join between daily stock prices (`dsf`) and stock names (`stock_names`) on the permanent security identifier `permno`.
- Each daily row is matched with the ticker and name valid on that date, by filtering based on the validity interval `[namedt, nameenddt_eff]`.
- Handles securities with multiple name intervals by keeping only the ticker/name valid at each trading date.
- Results in an enriched DataFrame `df_prices` with daily stock data including corresponding ticker and name information.


4.3 post join qualitycheck and cleaning

- Validates uniqueness on `(permno, date)` key to ensure no duplicates exist after merging.
- Reports the percentage of rows missing ticker data, highlighting potential coverage gaps in name history.
- Warns about unexpected data patterns such as near-zero adjusted prices or negative market capitalization.
- When the parameter `remove_unclean_permnos=True` is set, all records corresponding to `permno`s that have any missing ticker rows are removed.
- This cleaning helps exclude securities with incomplete or questionable name data, ensuring downstream analysis uses reliable data.

*Purpose:* Ensures consistent stock identifier mapping during joins.

---

## 5. Joining Data

- Combine daily stock prices (`df_prices`) with stock names using an as-of join aligned by `permno` and valid date intervals.
- Add Fama-French daily market factors (`ff`) by joining on date to provide market risk context.
- Integrate IBES consensus summary data (`ibes_statsumu`) prepared for daily matches to enrich fundamentals.
- Append IBES actual EPS announcements (`ibes_actu`) also prepared for daily granularity.
  
*Purpose:* To create a comprehensive dataset combining daily price, market risk, and fundamental data for each stock-date.

---

## 6. Post-Join Quality Assurance

- Perform data integrity checks after each join: ensure uniqueness of indices to avoid duplicate rows.
- Report on null values and data coverage for critical columns such as ticker symbols, prices, and market caps.
- Warn on anomalies like near-zero or negative adjusted prices or market caps to flag poor data quality.
  
*Purpose:* Maintain dataset consistency and detect any issues early to prevent data leakage or model bias.

---

## 7. Missing Value Imputation

- Apply forward-fill and backward-fill imputation within each `permno` group to fill temporal gaps in numeric features.
- Add optional columns to track source dates of forward-filled values for transparency.
- Remove leading rows with NaNs that cannot be forward filled to maintain a clean time series.
- Warn if any missing values remain post-imputation for further attention.
  
*Purpose:* Ensure continuous, gap-free time series data for each stock required by machine learning models.

---

## 8. Feature Engineering

- Augment the dataset by computing technical indicators such as moving averages and momentum from prices.
- Remove rows with all NaNs after indicator addition to keep data quality high.
- Assemble the final modeling matrix:
  - Target variable \( Y \) is set as the next-day log return of the adjusted closing price.
  - Lag relevant market factors (Fama-French) and fundamental actuals (IBES EPS) appropriately to avoid lookahead bias.
  - Select core features including prices, volumes, market caps, IBES consensus variables, and lagged factors.
  - Optionally drop rows missing essential features to keep the dataset robust.
  
*Purpose:* Build a temporally aligned feature matrix optimized for predictive modeling of next-day returns.

---

## 9. Final Clean-Up and Output

- Reset the DataFrame index, typically dropping the ‘date’ level index to flatten the structure for modeling.
- Print informative diagnostics about shape, index levels, columns, and missing data status of the final matrix.
- Return the fully prepared modeling DataFrame ready for machine learning workflows.
  
*Purpose:* Provide a clean, well-structured dataset ensuring transparency and reliability for downstream tasks.

---

# Explanation of Key Cleaning Actions

| Action                            | Purpose / Reason                                                                                           |
|----------------------------------|-----------------------------------------------------------------------------------------------------------|
| Remove rows with zero adjustment factors | These rows produce invalid adjusted prices and distort returns calculations.                               |
| Remove stocks with >1% negative prices   | Excessive negative prices indicate problematic data, so removing these stocks reduces noise.              |
| Convert prices to absolute values         | Ensures all price data is positive and consistent after cleaning.                                         |
| Forward and backward fill missing data    | Fills gaps in time series allowing continuous modeling and feature engineering.                           |
| Lag factors and actual results             | Prevents lookahead bias by aligning predictors with correct target time frames.                           |

---

