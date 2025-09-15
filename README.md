# MATH60610A-portfolio-backtesting

## Installation

Clone the repo. Create a virtual python environment `venv` and run

```bash
pip install -r requirements.txt
```

Data already exists. You can "Run all" in `main.ipynb` to train the strategy. Equity curves and statistics are found in
`main.ipynb` and in `out/yats_tearsheet.html` which is can be opened with your browser.

## Useful commands

```bash
pre-commit run --all-files
```

```bash
git ls-files | grep -vE '\.(parquet|html)$' | xargs cat | wc -l
```
