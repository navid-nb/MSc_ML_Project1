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

## 3. DSF Cleaning

- **Remove stocks where adjustment factors contain zeros.**  
    Adjustment factors like `cfacpr` and `cfacshr` should be strictly positive; zeros indicate bad or incomplete data which should be excluded.

- **Handle negative prices in DSF:**  
  In WRDS CRSP Daily Stock File, negative price values are not errors and do not indicate actual negative prices. Instead, a negative sign indicates that the reported price is a bid/ask average rather than an official closing price. For example, this can occur on days when no trades actually happened. This means such values are not fundamentally invalid and can contain useful information about price levels on days without closing prices. However, if a stock has an excessive fraction of negative prices exceeding a threshold (1%), it usually indicates poor data quality for that stock, so those stocks are removed to reduce noise in modeling.


- **Convert remaining price values to absolute values**  
  After cleaning, prices are converted to absolute values so that all price data is consistent and positive for further analysis.

- **Recalculate adjusted prices, shares, and market caps** using cleaned data.


*Purpose:* Ensure financial data is clean, consistent, and correctly adjusted for splits/dividends.

---

## 4. Stock Names Quality Checks

- Verify date types and check for overlapping validity windows in stock naming history.
- Overlaps may cause ambiguous ticker mappings and are flagged.

*Purpose:* Ensures consistent stock identifier mapping during joins.

---

## 5. Joining Data

- **Join DSF prices with stock names** via an as-of merge aligned on date with validity windows.
- **Join Fama-French factors (FF)** on date to enrich stock prices with market risk factors.
- **Join IBES consensus summary (EPS statsumu)** prepared to one row per ticker-date.
- **Join IBES actual EPS announcements (ibes_actu)** likewise prepared and cleaned.

*Purpose:* Merge all relevant financial and fundamental data into a single price-firm-date DataFrame for modeling.

---

## 6. Post-Join Quality Assurance

- After each join step, perform:
  - Uniqueness checks to avoid duplicates inflating rows
  - Null and coverage reporting for critical columns like tickers and price
  - Warning on near-zero or negative adjusted prices and market caps

*Purpose:* Maintain data consistency and early detection of bad merges or gaps.

---

## 7. Missing Value Imputation

- Perform **forward-fill and backward-fill grouped by permno** to impute missing numeric features for continuous modeling.
- Use a **final fallback imputation strategy (median)** for any remaining missing values.

*Purpose:* Fill temporal gaps in data while maintaining time series continuity for each stock.

---

## 8. Feature Engineering

- Add **technical indicators** (e.g., moving averages, momentum) based on prices.
- Build the **model matrix** to predict next-day returns:
  - Target variable \(Y\) is the log return of adjusted prices shifted by one day.
  - Create lagged factors (Fama-French) and lagged actual EPS variables.
  - Drop raw unlagged factor columns to prevent data leakage.
  - Select a core feature set of prices, volumes, market caps, and IBES consensus fields.

*Purpose:* Prepare final feature set with necessary predictors for the modeling task, aligned temporally to prevent lookahead bias.

---

## 9. Final Clean-Up and Return

- Controlled drop of rows missing target or core features (e.g., adjusted price, market cap, returns).
- Reset index and print diagnostic information including missing value reports.

*Purpose:* Return a clean, fully prepared DataFrame ready for machine learning models.

---

# Explanation of Key Cleaning Actions

| Action                          | Why?                                                       |
|-------------------------------|------------------------------------------------------------|
| Remove rows with zero adjustment factors | Zero adjustment factors indicate corrupt data; dropping ensures valid adjusted prices. |
| Remove stocks with >1% negative prices | Negative prices are suspicious; removal prevents erroneous model training. |
| Absolute value of prices       | After cleaning, prices should be positive; negative values are data errors or corrections. |
| Forward/backward-fill missing data | Ensures continuous time-series for each stock without gaps. |
| Lagging factors and actuals   | Prevents lookahead bias by aligning features correctly with target return dates. |

