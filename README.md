# MATH60610A Portfolio Backtesting (Quick Start)

**Everything runs fully offline.** No internet is required.
You can disable your WiFi (assuming your pre-installed
python version is supported).

## Requirement
- **Python 3.10 – 3.13**. The installer enforces this supported range.
  - If python is not installed, you can install it using an installer on 
    windows (see [python.org](python.org)) or using homebrew on MacOS. 
    On Mac, run `brew update` and then run `brew search python` and 
    `brew install python@3.10` (3.11, 3.12, 3.13 are also supported).
- Supports x86-64 and ARM (i.e. M1 Mac, Intel Mac, and Windows).
- GPU is not needed.

## What's Included (offline bundle)
- `wheels/`: prebuilt Python wheels for offline install  
- `requirements.lock` + `requirements[*].txt`: pinned to the included wheels  
- `data/`: Parquet snapshot (WRDS/YFinance) ready to use  
- `run_install_packages.py`: Creates venv and installs from `wheels/`  
- `run_data.py`: Downloads the data from WRDS and YFinance  
- `run_strategy.py`: Runs the strategy and writes to `outputs/`  
- `functions/`, `outputs/`, `docs/`: code, results, and documentation

## Quick Start (no venv activation needed)

> Before you start: Check your Python version by running `python3 --version`.

Navigate to project root directory and run:

**macOS/Linux**
```bash
python3 run_install_packages.py --offline-only
./.venv/bin/python run_strategy.py
```

**Windows (PowerShell or CMD)**
```shell
py -3 run_install_packages.py
.\.venv\Scripts\python.exe run_strategy.py
```

> Tip: If you have multiple Python versions installed, you can force one:
> `python3 run_install_packages.py --offline-only --python python3.10`  or  `py -3.10 run_install_packages.py --python py -3.10`

`run_strategy.py` will auto-run `run_data` and populate the `data` folder 
if no data is detected or if the user wants a new batch of data. If you wish 
to run `run_data` and populate the data folder (if not empty), you can run:
```bash
./.venv/bin/python run_data.py
```
then run 
```bash
./.venv/bin/python run_strategy.py
```
which will catch the newly added data

## Data

This deliverable comes with a **Parquet snapshot** under `data/` so everything runs **offline**.  
The snapshot aggregates five WRDS datasets plus a small Yahoo Finance macro set, then applies the
cleaning/join steps shown below. If you prefer to **rebuild/refresh** the snapshot, follow the
optional online steps at the end (requires valid WRDS access and internet).

### 1) Using the snapshot (default, offline)

- **Default run** (`run_strategy.py`) uses the included `data/` snapshot (no network required).  
- If no snapshot is found, the code calls into `run_data` to populate `data/` automatically.

---

### 2) (Optional) Rebuild/refresh the snapshot online

1. Ensure you have **WRDS credentials**.
2. Simply run`run_data.py`
3. This will create a new run folder within `data/` containing new Parquet files

---

### 3) What's in the snapshot

**WRDS (via HEC's WRDS license):**
- **CRSP – Daily Stock File (`crsp.dsf`)** -> `dsf.parquet`  
  Extracted by `functions/migrations/001_base_extract.sql`.
- **Fama–French Daily Factors (`ff.factors_daily`)** -> `ff.parquet`  
  Extracted by `functions/migrations/002_ff_factors.sql`.
- **IBES – U.S. EPS Consensus (statsumu) (`ibes.statsumu_epsus`)** -> `ibes_stats.parquet`  
  Extracted by `functions/migrations/003_ibes_statsumu.sql`.
- **IBES – U.S. EPS Actuals (`ibes.actu_epsus`)** -> `ibes_act.parquet`  
  Extracted by `functions/migrations/004_ibes_actu.sql`.

**Yahoo Finance (via `yfinance`):**
- **Macro & market context:** `^VIX, ^VXN, ^OVX, ^GVZ, ^GSPC, ^IXIC, ^RUT, XLK, XLF, XLE, XLV, XLI`  
  Saved to `common_features.parquet` by `common_features_extract(...)` in
  `functions/helpers/data_extraction.py`.

---

### 4) How the snapshot is built

The pipeline is orchestrated by `run_data.py` using modularized helpers in `functions/helpers/`:

1. **Raw extraction (WRDS) -> Parquet files**  
   `wrds_extract_raw(...)` runs the SQL files above and writes artifacts into a timestamped run
   folder under `data/` (e.g., `data/run_YYYYMMDD_HHMMSS/`).
2. **Macro features (Yahoo Finance) -> Parquet**  
   `common_features_extract(start_date, end_date, ...)` downloads the tickers listed above and saves
   `common_features.parquet` in the same run folder.
3. **Indexing & Quality assurance (QA)**  
   - `ensure_index(..., ['permno','date'])`, `pre_qa_dsf(...)` validate keys & dtypes.  
   - `clean_dsf(...)` removes entities with zero adjustment factors (`cfacpr/cfacshr`) and excessive
     negative prices, then recomputes adjusted price/shares/market cap.
4. **Ticker filtering and cross-dataset alignment**  
   `filter_by_tickers(...)` focuses on specified stock universe, while 
   `filter_by_tickers_and_permno_pairs(...)` ensures FF and IBES data only includes 
   permno-date pairs present in the main DSF dataset.
5. **Sequential data integration with quality assurance**  
   - **Fama-French**: `pre_qa_ff(...)` -> `join_prices_with_ff(...)` -> `post_join_qa_prices_with_ff(...)`  
   - **IBES Consensus**: `pre_qa_ibes_statsumu(...)` -> `prepare_ibes_for_daily_merge(...)` -> `join_prices_with_ibes(...)` -> `post_join_qa_prices_with_ibes(...)`  
   - **IBES Actuals**: `pre_qa_ibes_actu(...)` -> `prepare_ibes_actu_for_daily_merge(...)` -> `join_prices_with_ibes_actu(...)` -> `post_join_qa_prices_with_ibes_actu(...)`
6. **Yahoo Finance market context integration**  
   `join_prices_with_common_features(...)` attaches volatility indices (VIX, VXN, OVX, GVZ), 
   market indices (S&P 500, NASDAQ, Russell 2000), and sector ETFs (XLK, XLF, XLE, XLV, XLI) 
   to **every permno** per trading date for cross-asset feature generation.
7. **Imputation & alignment**  
   - `forward_fill_and_remove_initial_nans(...)` forward-fills within each `permno` and drops leading
     rows that remain incomplete.  
   - `align_and_fill_dates_across_tickers(...)` aligns all entities to a common trading-day calendar
     and validates identical row counts per entity.
8. **Feature engineering**  
   `feature_engineering.py` adds technical indicators (`pandas_ta`), macro ratios (e.g.,
   `ratio_^VIX_^GSPC`, sector-vs-SPX), log returns, lags, and the target(s) (`adj_prc_logret_lead1`).
   `adj_prc_logret_lead1` is the next-day log return of a stock’s adjusted price which corrects for 
   stock splits and similar share-changing events but does not add cash dividends.
9. **Final model matrix**  
   `build_final_matrix(...)` selects the modeling features & target(s), logs what was dropped, and
   returns the matrix indexed by `['permno','date']`.

---



