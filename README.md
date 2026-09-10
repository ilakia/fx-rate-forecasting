# FX Rate Forecasting

Forecasting daily FX close prices (starting with EUR/USD, later extended
to GBP/USD and USD/JPY) using a progression of time-series methods -
classical statistical (ARIMA/SARIMA), gradient-boosted trees (XGBoost),
and deep learning (LSTM) - with an emphasis on validating every result
before trusting it, and being explicit about where forecasting FX is
genuinely hard.

*This is an early-stage build. Sections below marked "(to come)" will be
filled in as later phases land - see `DECISIONS_AND_ISSUES_LOG.md` for
the full running log of what's been done and why.*

## Project Overview

Foreign exchange rates are notoriously close to a random walk - unlike,
say, retail demand or energy load, there's no strong seasonal or
structural signal guaranteeing a model can beat a naive "tomorrow = today"
baseline. This project takes that seriously: every model built here will
be benchmarked against a naive persistence baseline, and if a fancier
model can't beat it, that's reported as a real (and realistic) finding,
not hidden.

**Currency pairs (planned):** EUR/USD (current phase), GBP/USD, USD/JPY.

## Key Finding (EUR/USD)

**Three independent modeling approaches - linear statistical (ARIMA),
nonlinear tabular (XGBoost, searched across 32 hyperparameter
configurations), and deep sequential (LSTM) - all confirm that EUR/USD
daily returns show no exploitable structure beyond the naive "no change"
baseline.** RMSE/MAE match the naive baseline to 3-4 significant figures
across every model, and directional accuracy never moves meaningfully
off a coin flip (49-52%) on either the validation or test period. This
is consistent with weak-form market efficiency in a highly liquid FX
pair, and it's reported here as a real, defensible research outcome -
not a shortfall of the project. See "Results" below for the full
comparison table and `DECISIONS_AND_ISSUES_LOG.md` (#12, #16, #20) for
the complete reasoning behind why this is a finding worth trusting
rather than a search that simply hasn't found the right model yet.

## Data Source

**Primary source: [Frankfurter API](https://frankfurter.dev)** (`api.frankfurter.dev/v1`)
- Free, no API key, no signup, no credit card at any tier.
- Republishes the European Central Bank's official daily reference rates
  (published ~16:00 CET each TARGET2 business day) - verified byte-for-byte
  identical to a direct pull from the ECB's own Statistical Data Warehouse
  for a sample of test dates.
- Daily granularity, full history available from 1999-01-04 (the start of
  the Euro) through the present.
- Open source (github.com/lineofflight/frankfurter); underlying data
  license/attribution: European Central Bank.

Two other candidates were evaluated and rejected - **exchangerate.host**
(now requires an API key and, per its current host apilayer, a credit
card even to sign up) and a **direct ECB SDW pull** (works fine, same
data, but a more brittle SDMX format to maintain across three currency
pairs). Full reasoning for both in `DECISIONS_AND_ISSUES_LOG.md` (#1, #2).

**Currently pulled data:** EUR/USD daily close, 1999-01-04 to 2026-09-10
(7,090 rows). See `data/processed/eurusd_daily.csv`.

**Gap handling:** FX markets don't trade on weekends or Eurosystem
(TARGET2) holidays. These calendar gaps are real and expected - they are
**not** imputed, forward-filled, or otherwise treated as missing data.
Every gap longer than a normal weekend was checked and lines up with a
known holiday closure (Christmas/New Year, Good Friday/Easter Monday).
See `DECISIONS_AND_ISSUES_LOG.md` (#4) for the full data quality check.

## Methodology

**Target variable: daily log return**, `r_t = ln(P_t / P_{t-1})`, not raw
price. An Augmented Dickey-Fuller test confirmed the raw close series is
non-stationary (p = 0.32) while the log return series is strongly
stationary (p < 1e-6) - see `DECISIONS_AND_ISSUES_LOG.md` #5 for the full
test output. This makes the naive baseline every model must beat
completely unambiguous: **predict r_t = 0** (the random-walk hypothesis).
The raw close series is kept throughout so return predictions can be
converted back to price levels for plots and directional-accuracy checks.

**Feature set** (`src/feature_engineering.py`), shared across the ML/DL
models:
- `lag_1` .. `lag_5` - the previous 5 trading days' log returns (one
  trading week; FX return autocorrelation is weak beyond a few days for
  a liquid major pair, so this isn't trimmed shorter or extended further
  without evidence it helps).
- `roll_mean_5` / `roll_std_5` and `roll_mean_20` / `roll_std_20` -
  rolling return statistics over 1 week and ~1 trading month. The rolling
  std terms are a realized-volatility proxy, motivated by visible
  volatility clustering in the return series (calmer 2015-2019, choppier
  2008-2009 and 2022 - see `figures/eurusd_log_returns.png`).
- `dow_mon` .. `dow_fri` - one-hot day-of-week, included as a candidate
  feature on the strength of known FX day-of-week liquidity patterns;
  left to feature-importance analysis at training time to confirm or
  reject, not judged here.

All rolling/lag features are computed with `.shift(1)` applied **before**
any `.rolling()` call, so a feature at row t only ever sees r_{t-1} and
earlier - verified with an explicit runtime assertion in
`feature_engineering.py`, not just asserted in prose. See
`DECISIONS_AND_ISSUES_LOG.md` #6 for the full leakage-prevention writeup.

**Train / validation / test split** - chronological (never shuffled),
at calendar-year boundaries:

| Split | Date range | Rows |
|---|---|---|
| Train | 1999-02-02 -> 2019-12-31 | 5,354 |
| Validation | 2020-01-02 -> 2023-12-29 | 1,027 |
| Test | 2024-01-02 -> 2026-09-10 | 688 |

Feature scaling is fit only on the training split and applied (not
re-fit) to validation/test; the fitted scaler is persisted to
`data/processed/eurusd_feature_scaler.joblib`. Full reasoning for the
split boundaries (and why the validation window deliberately spans the
COVID crash and the 2022 rate-hike shock) is in
`DECISIONS_AND_ISSUES_LOG.md` #7.

**Shared evaluation module** (`src/evaluation/evaluate.py`) - every model
(naive, ARIMA now; XGBoost/LSTM next) is scored by identical logic:
RMSE/MAE on the log-return scale, RMSE/MAE reconstructed to price level
using the *true* previous price at each step (not a chained forecast, to
avoid compounding error over a long holdout), and directional accuracy
with an explicit tie policy (a day is excluded if either the true or
predicted return is exactly 0.0 - see `DECISIONS_AND_ISSUES_LOG.md` #9).

**Models built so far:**
1. **Naive persistence baseline** (`src/models/naive_baseline.py`) -
   predicts r_t = 0 every day. The mandatory reference point every other
   model must be honestly compared against.
2. **ARIMA** (`src/models/arima_sarima.py`) - order selected via ACF/PACF
   inspection (`notebooks/03_acf_pacf_check.py`) plus an AIC/BIC grid
   search on the training split only (pmdarima's `auto_arima` was
   attempted first but fails to import in this environment due to a
   numpy 2.0.2 ABI incompatibility - see `DECISIONS_AND_ISSUES_LOG.md`
   #8). Evaluated via one-step-ahead **walk-forward** forecasting across
   validation + test (refitting every 20 trading days, extending state
   without refitting in between).
3. **SARIMA** (weekly seasonal terms) - tested and **rejected** on
   in-sample AIC evidence before spending walk-forward compute on it; see
   `DECISIONS_AND_ISSUES_LOG.md` #11.

4. **XGBoost** (`src/models/xgboost_model.py`) - trained on the raw
   (unscaled) feature columns from Stage 2. Hyperparameters selected via
   a 32-combination grid search over regularization-relevant parameters
   (max_depth, min_child_weight, subsample, colsample_bytree,
   learning_rate), each trained with early stopping on validation RMSE.
   Evaluated with a single fit + one-shot predict (not walk-forward) -
   a deliberate, justified difference from ARIMA's evaluation, not an
   inconsistency; see `DECISIONS_AND_ISSUES_LOG.md` #13.
5. **LSTM** (`src/models/lstm_model.py`, PyTorch - TensorFlow's pip wheel
   requires AVX instructions unavailable under Rosetta 2 on this Apple
   Silicon Mac's x86_64 venv, see `DECISIONS_AND_ISSUES_LOG.md` #17) -
   a small, deliberately-regularized single-layer LSTM (16 hidden units,
   0.2 dropout, 2,065 trainable parameters) over 20-day sequences of the
   *scaled* Stage 2 features, trained with early stopping on validation
   loss. Sequences are built independently per split (never crossing a
   train/val/test boundary) with an explicit runtime leakage check; see
   `DECISIONS_AND_ISSUES_LOG.md` #18-#19.

## Results

Naive, ARIMA/SARIMA, XGBoost, and LSTM - the complete planned set of
model families for EUR/USD.

ACF/PACF inspection of training log returns showed no lags meaningfully
outside the 95% significance band across 30 lags - visually close to
white noise. The resulting AIC/BIC grid search selected **ARIMA(0,0,0)**
(a plain constant-mean model, no AR/MA terms) as the best order. A
weekly-seasonal SARIMA variant was tested and rejected (AIC got slightly
*worse* with seasonal terms added). XGBoost's 32-combination hyperparameter
grid search **independently converged to `best_iteration=0` in every
single configuration** - the first tree already minimized validation
RMSE, and every additional tree made it worse, regardless of
regularization settings.

| Model | Period | RMSE (return) | MAE (return) | RMSE (price) | MAE (price) | Directional accuracy |
|---|---|---|---|---|---|---|
| Naive | val | 0.004976 | 0.003664 | 0.005412 | 0.004035 | undefined (100% ties) |
| ARIMA(0,0,0) | val | 0.004976 | 0.003664 | 0.005412 | 0.004036 | 49.2% |
| XGBoost | val | 0.004976 | 0.003664 | 0.005412 | 0.004035 | 51.2% |
| LSTM | val | 0.005003 | 0.003697 | 0.005441 | 0.004071 | 49.4% |
| Naive | test | 0.004212 | 0.002980 | 0.004687 | 0.003329 | undefined (100% ties) |
| ARIMA(0,0,0) | test | 0.004212 | 0.002981 | 0.004688 | 0.003329 | 51.0% |
| XGBoost | test | 0.004212 | 0.002980 | 0.004687 | 0.003329 | 51.6% |
| LSTM | test | 0.004282 | 0.003032 | 0.004765 | 0.003386 | 50.6% |

*(LSTM's val/test row counts are 1,007/668, not 1,027/688 - a 20-day
sequence warm-up is dropped per split, see `DECISIONS_AND_ISSUES_LOG.md`
#18. Not a systematically different period, so this doesn't explain
LSTM's small RMSE overhead.)*

**Honest read: none of the three models beats the naive baseline.**
RMSE/MAE match across all four rows to 3-4 significant figures on both
periods (LSTM runs marginally higher, not lower - the expected footprint
of a model with free parameters that didn't find real signal, rather
than an improvement), and directional accuracy never moves meaningfully
off a coin flip (49-52%) for any model. This is now **three independent
confirmations** of the same finding, across three structurally different
model families: linear (ARIMA), nonlinear tabular (XGBoost - 32/32
hyperparameter configurations converged to the same null result), and
deep sequential (LSTM - training/validation loss converged to a flat,
non-diverging plateau by ~epoch 15-20, see
`figures/lstm_loss_curve.png`). None of the three search/tuning
processes was adjusted after seeing validation or test results.

**Feature importance (XGBoost, gain-based, restricted to the actually-
deployed tree - see `DECISIONS_AND_ISSUES_LOG.md` #14 for why this
restriction mattered):** only 3 of 14 features have any nonzero gain at
all - `roll_std_20` (the dominant one), `lag_1`, and `lag_5` - and the
total gain across all three is tiny. **Day-of-week features show exactly
zero importance** - the known FX day-of-week liquidity pattern does not
translate into next-day return predictability that this model could
exploit. The rolling-volatility feature being the most (if barely) used
one is a small, genuinely interesting hint - even though the model
overall doesn't beat naive, it's suggestive that realized volatility
might carry more information than direction does, consistent with the
volatility clustering visible in `figures/eurusd_log_returns.png`.
Full writeup: `DECISIONS_AND_ISSUES_LOG.md` #13-#20.

This is now the complete EUR/USD model comparison - see "Key Finding"
above for the headline takeaway. The next stage extends this same
pipeline (data pull, features, all four models, this evaluation
framework) to GBP/USD and USD/JPY, to see whether weak-form efficiency
holds as consistently on other liquid major pairs.

## Limitations & Honest Findings

- **Only EUR/USD has been pulled so far.** GBP/USD and USD/JPY are planned
  but not yet built - the data pull script (`src/data_pull.py`) is
  parameterized by `--base`/`--quote` specifically so the same logic
  reapplies without rewriting.
- **FX is close to a random walk, and three independent models confirm
  it.** ARIMA (linear, data-driven order search), XGBoost (nonlinear,
  32-configuration regularized grid search), and LSTM (deep sequential,
  deliberately small and regularized) all found no exploitable
  structure in EUR/USD daily returns and could not beat the naive "no
  change" baseline. This project
  is explicitly designed to report this honestly rather than keep tuning
  until a better-looking number appears; see `DECISIONS_AND_ISSUES_LOG.md`
  #12, #16, and #20 for the full reasoning on why this is a legitimate
  finding, not a failure.
- Full engineering log of every decision, issue, and rejected alternative
  is kept in `DECISIONS_AND_ISSUES_LOG.md`.

## How to Reproduce

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# On Apple Silicon with an x86_64 (Rosetta) Python, `torch` from PyPI may
# need the CPU-specific index instead - see DECISIONS_AND_ISSUES_LOG.md #17:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

# Pull EUR/USD daily history (defaults to 1999-01-04 -> today)
python src/data_pull.py --base EUR --quote USD

# Run the exploratory data quality checks + plot
python notebooks/01_initial_exploration.py

# Compute log returns, run the ADF stationarity test, plot returns
python notebooks/02_target_and_stationarity.py

# Build features, run the leakage check, and produce the train/val/test split
python src/feature_engineering.py --input data/processed/eurusd_daily.csv --prefix eurusd

# ACF/PACF inspection to inform ARIMA order selection
python notebooks/03_acf_pacf_check.py

# Naive baseline, ARIMA (order search + walk-forward), SARIMA seasonal check
python src/models/naive_baseline.py
python src/models/arima_sarima.py
python src/models/sarima_check.py

# XGBoost (hyperparameter grid search + feature importance)
python src/models/xgboost_model.py

# LSTM (sequence construction + leakage check + training)
python src/models/lstm_model.py
```

## Repo Structure

```
fx-rate-forecasting/
├── README.md
├── DECISIONS_AND_ISSUES_LOG.md   # full engineering/methodology log
├── requirements.txt
├── data/
│   ├── raw/                      # untouched API responses
│   └── processed/                # cleaned daily series, feature tables, splits, scaler, model results
├── src/
│   ├── data_pull.py              # reusable pull script, any currency pair
│   ├── feature_engineering.py    # reusable feature/split builder, any pair
│   ├── evaluation/
│   │   └── evaluate.py           # shared scoring module, used by every model
│   └── models/
│       ├── naive_baseline.py     # predict r_t = 0
│       ├── arima_sarima.py       # ARIMA order search + walk-forward evaluation
│       ├── sarima_check.py       # weekly-seasonal AIC pre-check (rejected)
│       ├── xgboost_model.py      # hyperparameter grid search + feature importance
│       └── lstm_model.py         # sequence construction, leakage check, PyTorch LSTM
├── notebooks/
│   ├── 01_initial_exploration.py       # gap/flat-line/jump checks + plot
│   ├── 02_target_and_stationarity.py   # log return, ADF test, return plot
│   └── 03_acf_pacf_check.py            # ACF/PACF inspection for ARIMA order
└── figures/
    ├── eurusd_1999_2026.png
    ├── eurusd_log_returns.png
    ├── eurusd_acf_pacf.png
    └── lstm_loss_curve.png
```
