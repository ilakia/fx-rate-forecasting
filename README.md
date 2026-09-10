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

**Models planned next:**
4. XGBoost.
5. LSTM.

## Results

*Partial - naive baseline and ARIMA/SARIMA only. XGBoost and LSTM are not
yet built; this table will grow, not get overwritten, as they land.*

ACF/PACF inspection of training log returns showed no lags meaningfully
outside the 95% significance band across 30 lags - visually close to
white noise. The resulting AIC/BIC grid search selected **ARIMA(0,0,0)**
(a plain constant-mean model, no AR/MA terms) as the best order. A
weekly-seasonal SARIMA variant was tested and rejected (AIC got slightly
*worse* with seasonal terms added).

| Model | Period | RMSE (return) | MAE (return) | RMSE (price) | MAE (price) | Directional accuracy |
|---|---|---|---|---|---|---|
| Naive | val | 0.004976 | 0.003664 | 0.005412 | 0.004035 | undefined (100% ties) |
| ARIMA(0,0,0) | val | 0.004976 | 0.003664 | 0.005412 | 0.004036 | 49.2% |
| Naive | test | 0.004212 | 0.002980 | 0.004687 | 0.003329 | undefined (100% ties) |
| ARIMA(0,0,0) | test | 0.004212 | 0.002981 | 0.004688 | 0.003329 | 51.0% |

**Honest read: ARIMA does not beat the naive baseline.** RMSE/MAE match
to 3-4 significant figures, and ARIMA's directional accuracy sits right
on a coin flip (49-51%) with no consistent edge across the two periods.
This is expected, not a project shortfall: ARIMA(0,0,0) is mechanically
just the training-mean return (a tiny, near-zero constant), so it makes
almost the same prediction as naive's flat zero every day. The order
search was run once against the training split only (never re-tuned
against validation or test results), so this isn't a search that "hasn't
found the right order yet" - the ACF/PACF evidence gives a principled
reason to expect no linear autocorrelation structure exists to find.
Full writeup: `DECISIONS_AND_ISSUES_LOG.md` #10-#12.

This sets the real bar for what comes next: XGBoost and LSTM are
nonlinear and could in principle find structure a linear ARIMA can't -
if they *also* fail to beat naive, that's a stronger finding about this
specific problem, not a weaker one.

## Limitations & Honest Findings

- **Only EUR/USD has been pulled so far.** GBP/USD and USD/JPY are planned
  but not yet built - the data pull script (`src/data_pull.py`) is
  parameterized by `--base`/`--quote` specifically so the same logic
  reapplies without rewriting.
- **FX is close to a random walk, and the first real model confirms it.**
  ARIMA - a linear time-series model with a data-driven order search -
  found no exploitable autocorrelation structure in EUR/USD daily
  returns and could not beat the naive "no change" baseline. This project
  is explicitly designed to report this honestly rather than keep tuning
  until a better-looking number appears; see `DECISIONS_AND_ISSUES_LOG.md`
  #12 for the full reasoning on why this is a legitimate finding, not a
  failure.
- Full engineering log of every decision, issue, and rejected alternative
  is kept in `DECISIONS_AND_ISSUES_LOG.md`.

## How to Reproduce

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

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
│       └── sarima_check.py       # weekly-seasonal AIC pre-check (rejected)
├── notebooks/
│   ├── 01_initial_exploration.py       # gap/flat-line/jump checks + plot
│   ├── 02_target_and_stationarity.py   # log return, ADF test, return plot
│   └── 03_acf_pacf_check.py            # ACF/PACF inspection for ARIMA order
└── figures/
    ├── eurusd_1999_2026.png
    ├── eurusd_log_returns.png
    └── eurusd_acf_pacf.png
```
