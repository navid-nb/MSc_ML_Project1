# ML in Finance — Long-Short Equity Strategy: "Yet Another Hedge Fund"

End-to-end systematic long-short equity strategy built on machine learning signal generation, ensemble confirmation, and portfolio optimization. The pipeline spans data ingestion from WRDS/Yahoo Finance, feature engineering, ElasticNet-regularized classification and regression models, dual-model signal confirmation, and backtesting across 9 allocation strategies on ~65 large-cap U.S. stocks (2010–2025).

---

## Strategy Overview

The strategy predicts next-day stock return direction and magnitude using two ML models trained on price, fundamental, and macro features. Only stocks where both models agree on direction enter the trading universe, reducing noise and false signals. The combined signal then feeds into multiple portfolio construction approaches benchmarked against each other.

---

## Pipeline & Methods

**1. Data Ingestion & Feature Engineering**
Pulled from four WRDS datasets (CRSP daily stock file, Fama-French daily factors, IBES EPS consensus, IBES EPS actuals) and Yahoo Finance macro series (VIX, VXN, OVX, GVZ, S&P 500, NASDAQ, Russell 2000, sector ETFs: XLK, XLF, XLE, XLV, XLI). Feature set includes:
- Technical indicators: RSI, MACD, ATR, ADX, Bollinger Bands (via `pandas-ta`)
- Cross-asset ratios: VIX/S&P 500, sector ETF vs. SPX
- Fama-French risk factors (Mkt-RF, SMB, HML)
- IBES EPS surprise and analyst revision signals
- Lagged log returns and momentum signals
- Target: next-day adjusted price log return (`adj_prc_logret_lead1`)

**2. Model 1 — Direction Prediction (Logistic Regression)**
Trained ElasticNet-penalized logistic regression (solver: SAGA) to classify next-day return direction (up/down). Hyperparameter tuning of l1_ratio and C via rolling cross-validation (10 folds, 60/20 train/validation split). Optimal: l1_ratio = 0.9, C = 0.1. Generates probability-of-up signal P(↑); stocks with P(↑) > 0.55 are long candidates, P(↑) < 0.45 are short candidates.

**3. Model 2 — Magnitude Prediction (Linear Regression)**
Trained ElasticNet-penalized regression to predict next-day log return magnitude. Hyperparameter tuning of l1_ratio and alpha. Optimal: l1_ratio = 0.5, α = 0.0001. Expected return above 0.1% → long signal; below −0.1% → short signal.

**4. Ensemble Signal Confirmation**
Both models must agree on direction to generate a confirmed trading signal. Disagreement → position excluded. This dual-confirmation filter reduces trade frequency but improves signal quality. Ensemble score = logistic score (scaled to [−1, +1]); zeroed for disagreements.

**5. Portfolio Construction — 9 Allocation Strategies**
Applied to the agreed long/short universe daily with 50/50 dollar neutrality and 10% max position cap:
- **A1** Equal-weighted (EW) — benchmark
- **A2** Rank-weighted (score-proportional weights)
- **A3** Top/bottom quantile (top 20% longs, bottom 20% shorts)
- **A4** Long-only threshold
- **A5** Inverse-volatility weighting
- **A6** Maximum Sharpe (mean-variance optimization via SLSQP)
- **A7** Risk parity / equal risk contribution (1/σ weights)
- **A8** Softmax allocation (exponential score mapping)
- **A9** Fractional Kelly criterion (μ/σ² with 25% Kelly fraction)

**6. Backtesting & Performance**
Rolling 65%/35% train/out-of-sample split. Metrics: annualized return, Sharpe ratio, Sortino ratio, max drawdown, win rate, equity curve. Out-of-sample evaluation is strictly enforced (models fitted only on in-sample data).

---

## Key Files

| File | Description |
|------|-------------|
| [`docs/Yet_Another_Hedge_Fund.pdf`](docs/Yet_Another_Hedge_Fund.pdf) | Full strategy report with methodology, results, and performance analysis |
| [`functions/pipeline_orchestrator.py`](functions/pipeline_orchestrator.py) | Main ML pipeline: feature selection, model training, signal generation, backtesting |
| [`functions/helpers/allocation_strategies.py`](functions/helpers/allocation_strategies.py) | All 9 portfolio allocation strategies (A1–A9) |
| [`functions/helpers/portfolio_backtest.py`](functions/helpers/portfolio_backtest.py) | Performance metrics: Sharpe, Sortino, max drawdown, equity curve |
| [`functions/helpers/feature_engineering.py`](functions/helpers/feature_engineering.py) | Technical indicators, cross-asset ratios, target construction |
| [`functions/helpers/data_cleanup.py`](functions/helpers/data_cleanup.py) | WRDS data QA, joins, forward-fill, alignment |
| [`run_strategy.py`](run_strategy.py) | Entry point — runs full pipeline end-to-end |
| [`docs/Key Metrics.png`](docs/Key%20Metrics.png) | Summary of out-of-sample strategy performance |
| [`docs/Strategy.png`](docs/Strategy.png) | Equity curve and allocation strategy comparison |

**Data:** Pre-built Parquet snapshot under `data/` (WRDS + Yahoo Finance). WRDS access required only to rebuild.

---

## Tools & Libraries

Python · scikit-learn (ElasticNet, LogisticRegression, Pipeline, GridSearchCV) · pandas-ta (technical indicators) · pandas · NumPy · SciPy (portfolio optimization) · WRDS · yfinance · Dagster (pipeline orchestration) · Docker / AWS ECR + EC2 (deployment)

---

## Setup

**Quick start (offline, no WRDS needed):**

```bash
# macOS/Linux
python3 run_install_packages.py
./.venv/bin/python run_strategy.py

# Windows
py -3 run_install_packages.py
.\.venv\Scripts\python.exe run_strategy.py
```

The pre-built `data/` Parquet snapshot is included. `run_strategy.py` automatically calls `run_data.py` if no data is found. To rebuild from WRDS, set your credentials and run `run_data.py` separately.

**Python 3.10–3.13 required. No GPU needed.**
