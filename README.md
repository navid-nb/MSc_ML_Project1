# MATH60610A Portfolio Backtesting (Quick Start)

**Everything runs fully offline.** No internet is required to:
- set up the virtual environment,
- load WRDS & Yahoo Finance data (snapshot included),
- run the backtest and generate outputs.

## Requirement
- **Python 3.10.\*** (e.g., 3.10.14). The installer enforces 3.10.

## What’s Included (offline bundle)
- `wheels/`: prebuilt Python wheels for offline install  
- `requirements.lock` + `requirements.txt`: pinned to the included wheels  
- `data/`: Parquet snapshot (WRDS/YFinance) ready to use  
- `run_install_packages.py`: creates venv and installs from `wheels/`  
- `run_data.py`: prepares/validates data using the snapshot  
- `run_strategy.py`: runs the backtest and writes to `outputs/`  
- `functions/`, `outputs/`, `docs/`: code, results, and documentation

## Quick Start (no venv activation needed)

**macOS/Linux**
```bash
python3.10 run_install_packages.py
./.venv/bin/python run_strategy.py
```

**Windows (PowerShell or CMD)**
```shell
py -3.10 run_install_packages.py
.\.venv\Scripts\python.exe run_strategy.py
```

`run_strategy.py` will run the functions in `run_data` and populate the `data` folder 
if no data is detected or if the user wants a new batch of data.
