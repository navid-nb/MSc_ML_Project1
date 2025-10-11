# MATH60610A Portfolio Backtesting (Quick Start)

**Everything runs fully offline.** No internet is required.
You can disable your WiFi (assuming your pre-installed
python version is supported).

## Requirement
- **Python 3.10 – 3.13**. The installer enforces this supported range.
- If python is not installed, you can install it using an installer on 
  windows or using homebrew on MacOS. On Mac, run `brew search python` 
  and then run `brew update` and `brew install python@3.10` (3.11, 3.12, 
  3.13 are also supported).
- Supports x86-64 and ARM (i.e. M1 Mac, Intel Mac, and Windows).
- GPU is not needed.

## What's Included (offline bundle)
- `wheels/`: prebuilt Python wheels for offline install  
- `requirements.lock` + `requirements.txt`: pinned to the included wheels  
- `data/`: Parquet snapshot (WRDS/YFinance) ready to use  
- `run_install_packages.py`: creates venv and installs from `wheels/`  
- `run_data.py`: prepares/validates data using the snapshot  
- `run_strategy.py`: runs the backtest and writes to `outputs/`  
- `functions/`, `outputs/`, `docs/`: code, results, and documentation

## Quick Start (no venv activation needed)

> Before you start: Check your Python version by running `python3 --version`.
> It must be between 3.10 and 3.13 (inclusive). Otherwise, install a supported version before proceeding.

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

`run_strategy.py` will run the functions in `run_data` and populate the `data` folder 
if no data is detected or if the user wants a new batch of data.

## Data

This deliverable comes with a **Parquet snapshot** under `data/` so everything runs **offline**.  
The snapshot aggregates five WRDS datasets plus a small Yahoo Finance macro set, then applies the
cleaning/join steps shown below. If you prefer to **rebuild/refresh** the snapshot, follow the
optional online steps at the end (requires valid WRDS access and internet).

### 1) What's in the snapshot

**WRDS (via HEC's WRDS license):**
- **CRSP – Daily Stock File (`crsp.dsf`)** -> `dsf.parquet`  
  Extracted by `functions/migrations/001_base_extract.sql`.
- **CRSP – Stock Names (`crsp.stocknames`)** -> `stocknames.parquet`  
  Extracted by `functions/migrations/002_crsp_names.sql`.
- **Fama–French Daily Factors (`ff.factors_daily`)** -> `ff.parquet`  
  Extracted by `functions/migrations/003_ff_factors.sql`.
- **IBES – U.S. EPS Consensus (statsumu) (`ibes.statsumu_epsus`)** -> `ibes_stats.parquet`  
  Extracted by `functions/migrations/004_ibes_statsumu.sql`.
- **IBES – U.S. EPS Actuals (`ibes.actu_epsus`)** -> `ibes_act.parquet`  
  Extracted by `functions/migrations/005_ibes_actu.sql`.

**Yahoo Finance (via `yfinance`):**
- **Macro & market context:** `^VIX, ^VXN, ^OVX, ^GVZ, ^GSPC, ^IXIC, ^RUT, XLK, XLF, XLE, XLV, XLI`  
  Saved to `common_features.parquet` by `common_features_extract(...)` in
  `functions/helpers/data_extraction.py`.

---

### 2) How the snapshot is built

The pipeline is orchestrated by `run_data.py` and helpers in `functions/helpers/`:

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
4. **Entity/ticker mapping (as-of join)**  
   `join_dsf_with_stocknames(...)` merges CRSP names to DSF as of each trading date, collapses
   overlapping intervals, and ensures no row inflation.
5. **Factor & fundamentals joins**  
   - `join_prices_with_ff(...)` (on `date`) + `post_join_qa_prices_with_ff(...)`.  
   - `prepare_ibes_for_daily_merge(...)` -> `join_prices_with_ibes(...)` (on `(ticker,date)`), then
     `post_join_qa_prices_with_ibes(...)`.  
   - `prepare_ibes_actu_for_daily_merge(...)` -> `join_prices_with_ibes_actu(...)` (on `(ticker,date)`),
     then `post_join_qa_prices_with_ibes_actu(...)`.
6. **Macro joins**  
   `join_prices_with_common_features(...)` attaches the Yahoo Finance columns to **every permno** per
   trading date.
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

### 3) Using the snapshot (default, offline)

- **Default run** (`run_strategy.py`) uses the included `data/` snapshot (no network required).  
- If no snapshot is found, the code calls into `run_data` to populate `data/` automatically.

---

### 4) (Optional) Rebuild/refresh the snapshot online

1. Ensure you have **WRDS credentials**.
2. In `run_data.py`, call `build_model_matrix_from_wrds(...)` with:
   - `wrds_user="YOUR_WRDS_USERNAME"`,
   - desired `start`, `end`, and `chunk_size`,  
   - `use_run="new"` to create a new `data/run_.../` folder.
3. The function will:
   - extract WRDS artifacts -> Parquet,
   - download Yahoo Finance macro features,
   - run all joins/QA/engineering,
   - print shape/coverage summaries for the resulting matrix.

You can subsequently point your experiments at the new run folder or keep using the snapshot for
fully offline grading.

**Fun fact:** The data extractor pulls **all securities defined by the SQL migrations** for the 
chosen date range (independent of your ticker list). On later runs, you can change the ticker 
list freely with `use_run="last"`, so no internet needed unless you change the date range.
