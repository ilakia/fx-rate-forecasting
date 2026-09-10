# FX Rate Forecasting

Forecasting daily FX close prices for EUR/USD, GBP/USD, and USD/JPY
using a progression of time-series methods - classical statistical
(ARIMA/SARIMA), gradient-boosted trees (XGBoost), and deep learning
(LSTM) - with an emphasis on validating every result before trusting it,
and being explicit about where forecasting FX is genuinely hard.

*Core build (EUR/USD deep dive plus GBP/USD and USD/JPY cross-pair
validation) is complete - see `DECISIONS_AND_ISSUES_LOG.md` for the full
running log of what's been done and why.*

## Project Overview

Foreign exchange rates are notoriously close to a random walk - unlike,
say, retail demand or energy load, there's no strong seasonal or
structural signal guaranteeing a model can beat a naive "tomorrow = today"
baseline. This project takes that seriously: every model built here is
benchmarked against a naive persistence baseline, and if a fancier model
can't beat it, that's reported as a real (and realistic) finding, not
hidden.

**Currency pairs:** EUR/USD (primary deep-dive), GBP/USD, USD/JPY
(cross-pair validation, same pipeline and standards).

## Key Finding

**Three independent modeling approaches - linear statistical (ARIMA),
nonlinear tabular (XGBoost), and deep sequential (LSTM) - confirm across
all three major currency pairs (EUR/USD, GBP/USD, USD/JPY) that daily
returns show no exploitable structure beyond the naive "no change"
baseline.** RMSE/MAE match the naive baseline to within ~2% for every
model, pair, and period (18 model x pair x period comparisons total, and
not one shows a real, test-surviving improvement), and directional
accuracy never reflects genuine skill once checked properly. This is
consistent with weak-form market efficiency across highly liquid FX
pairs, and it's reported here as a real, defensible research outcome -
not a shortfall of the project.

Two results looked like possible exceptions at first glance and were
each investigated before being ruled out - this is why the finding is
trustworthy rather than merely convenient:
- **GBP/USD's ARIMA order search selected ARIMA(2,0,2)**, not the
  trivial (0,0,0) every other pair converged to, with a "decisive" AIC
  margin. Walk-forward evaluation showed it performs marginally *worse*
  than naive on both validation and test - an in-sample fit that never
  translated into real forecasting skill (`DECISIONS_AND_ISSUES_LOG.md` #24).
- **USD/JPY's directional accuracy looked elevated for ARIMA/LSTM**
  (~55-57% on test, vs. the ~49-52% coin-flip range everywhere else).
  Investigation traced this to the test period's genuine 56.8%-positive-day
  trend (real, sustained USD/JPY strength) colliding with each model's
  own small, arbitrary constant-direction bias - confirmed by XGBoost's
  *opposite* bias scoring *below* 50% on the identical period via the
  identical mechanism (`DECISIONS_AND_ISSUES_LOG.md` #26).

See "Results" below for the full per-pair tables and the unified
cross-pair comparison, and `DECISIONS_AND_ISSUES_LOG.md` (#12, #16, #20,
#24-#27) for the complete reasoning.

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

**Currently pulled data:** all three pairs, daily close, 1999-01-04 to
2026-09-10 (7,090 rows each, identical calendar) - `eurusd_daily.csv`,
`gbpusd_daily.csv`, `usdjpy_daily.csv` in `data/processed/`.

**GBP/USD and USD/JPY are ECB-implied cross rates, not independently
quoted.** The ECB only publishes reference rates against EUR, so
Frankfurter derives GBP/USD as EUR/USD ÷ EUR/GBP (verified to match
Frankfurter's own directly-returned value to 4 decimal places on a
sample date) - standard, reliable methodology, but a real provenance
distinction from EUR/USD's directly-observed rate. See
`DECISIONS_AND_ISSUES_LOG.md` (#21).

**Gap handling:** FX markets don't trade on weekends or Eurosystem
(TARGET2) holidays. These calendar gaps are real and expected - they are
**not** imputed, forward-filled, or otherwise treated as missing data.
Every gap longer than a normal weekend was checked and lines up with a
known holiday closure (Christmas/New Year, Good Friday/Easter Monday) -
identical across all three pairs, since all three are published on the
same ECB schedule. See `DECISIONS_AND_ISSUES_LOG.md` (#4, #22) for the
full data quality checks, including GBP/USD's one genuine >5% single-day
move (2016-06-24, the Brexit referendum result day - confirmed real, not
a data defect).

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

**The entire pipeline above (target, features, split, evaluation module,
all four models) is reused unchanged for GBP/USD and USD/JPY** - every
script in `src/` and `notebooks/` takes a `--prefix` argument
(`eurusd`/`gbpusd`/`usdjpy`), and the same year-boundary split, feature
set, and model architectures apply identically to all three pairs. Each
pair's own ADF test, ARIMA order search, XGBoost hyperparameter search,
and LSTM training reran independently rather than reusing EUR/USD's
selected order/hyperparameters - see "Results" below and
`DECISIONS_AND_ISSUES_LOG.md` #21-#27 for the cross-pair findings.

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

### EUR/USD

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
`figures/eurusd_lstm_loss_curve.png`). None of the three search/tuning
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

### GBP/USD

Same pipeline, same standards, rerun independently - see
`DECISIONS_AND_ISSUES_LOG.md` #21-#24 for data quality and ADF detail.
GBP/USD's own AIC/BIC search selected **ARIMA(2,0,2)**, not the trivial
null every other pair converged to - investigated in #24 and found not
to survive out-of-sample evaluation (RMSE marginally worse than naive on
both val and test).

| Model | Period | RMSE (return) | MAE (return) | RMSE (price) | MAE (price) | Directional accuracy |
|---|---|---|---|---|---|---|
| Naive | val | 0.006109 | 0.004431 | 0.007592 | 0.005604 | undefined (100% ties) |
| ARIMA(2,0,2) | val | 0.006118 | 0.004451 | 0.007604 | 0.005630 | 48.2% |
| XGBoost | val | 0.006109 | 0.004432 | 0.007592 | 0.005605 | 49.4% |
| LSTM | val | 0.006151 | 0.004481 | 0.007643 | 0.005666 | 49.3% |
| Naive | test | 0.004247 | 0.003114 | 0.005548 | 0.004075 | undefined (100% ties) |
| ARIMA(2,0,2) | test | 0.004261 | 0.003123 | 0.005565 | 0.004087 | 52.9% |
| XGBoost | test | 0.004249 | 0.003116 | 0.005549 | 0.004078 | 47.9% |
| LSTM | test | 0.004315 | 0.003178 | 0.005639 | 0.004162 | 46.2% |

XGBoost again converged to `best_iteration=0` (trivial prediction).
Feature importance was again dominated by `roll_std_20` (38% of total
gain), with day-of-week contributing a small but nonzero 3.8% (driven
almost entirely by `dow_mon`) - still far too small to represent a
usable signal, and consistent with EUR/USD's near-zero day-of-week
result. GBP/USD's raw return series includes one genuine >5% single-day
move (the 2016 Brexit shock, entry #22) inside the training window;
none of the models beat naive despite that extra volatility being
available to learn from.

### USD/JPY

Same pipeline again. ADF and SARIMA checks both confirm the same
pattern as the other two pairs (`DECISIONS_AND_ISSUES_LOG.md` #23,
sarima rejected on AIC). XGBoost's search behaved differently here - 314
deployed trees rather than instant convergence, with a small validation
improvement that did not survive on test (entry #25).

| Model | Period | RMSE (return) | MAE (return) | RMSE (price) | MAE (price) | Directional accuracy |
|---|---|---|---|---|---|---|
| Naive | val | 0.006105 | 0.004060 | 0.789008 | 0.509513 | undefined (100% ties) |
| ARIMA(0,0,0) | val | 0.006105 | 0.004060 | 0.789041 | 0.509420 | 50.5% |
| XGBoost | val | 0.006096 | 0.004054 | 0.787635 | 0.508737 | 54.6% |
| LSTM | val | 0.006151 | 0.004085 | 0.795305 | 0.512891 | 54.8% |
| Naive | test | 0.006149 | 0.004280 | 0.927657 | 0.648169 | undefined (100% ties) |
| ARIMA(0,0,0) | test | 0.006149 | 0.004275 | 0.927666 | 0.647410 | 56.9% |
| XGBoost | test | 0.006153 | 0.004280 | 0.928287 | 0.648226 | 46.3% |
| LSTM | test | 0.006210 | 0.004246 | 0.938074 | 0.643751 | 56.2% |

**The directional-accuracy numbers here need the caveat spelled out in
`DECISIONS_AND_ISSUES_LOG.md` #26: they are not evidence of real skill.**
USD/JPY's test period (2024-2026) had 391 up-days vs. 296 down-days
(56.8% positive) - a genuine, well-known sustained uptrend. ARIMA's
walk-forward-updated constant predicted a **positive** return on all 688
test days (inheriting a positive drift from train+val history); a model
that always guesses the majority class in an imbalanced period scores
above 50% by construction, with zero real day-to-day skill required.
XGBoost's predictions skewed the **opposite** way (403 negative vs. 285
positive) and scored *below* 50% via the identical mechanism, in the
identical period - confirming this is about each model's arbitrary
constant-ish bias meeting a real trend, not about which model
"understands" USD/JPY better. RMSE - which doesn't care about class
balance - independently confirms no real skill: every model is within
~1% of naive on both periods.

### Cross-Pair Comparison

Unified table (`data/processed/cross_pair_comparison.csv`,
`src/cross_pair_comparison.py`) computing each model's RMSE ratio to
naive for every pair x period. Across all 3 pairs x 3 non-naive models x
2 periods = **18 comparisons, not one shows a ratio below 0.98**
(i.e. a real >2% RMSE improvement over naive):

| Pair | Model | Val ratio | Test ratio |
|---|---|---|---|
| EUR/USD | ARIMA | 1.0001 | 1.0001 |
| EUR/USD | XGBoost | 1.0000 | 1.0000 |
| EUR/USD | LSTM | 1.0055 | 1.0167 |
| GBP/USD | ARIMA | 1.0014 | 1.0031 |
| GBP/USD | XGBoost | 1.0000 | 1.0003 |
| GBP/USD | LSTM | 1.0069 | 1.0158 |
| USD/JPY | ARIMA | 1.0000 | 1.0000 |
| USD/JPY | XGBoost | 0.9987 | 1.0007 |
| USD/JPY | LSTM | 1.0076 | 1.0099 |

**The EUR/USD finding generalizes cleanly to all three major pairs.**
Three structurally different model families, applied identically across
three pairs with genuinely different macro histories (EUR/USD's steady
multi-year cycles, GBP/USD's Brexit shock, USD/JPY's sustained
2022-2025 uptrend), converge on the same answer: no exploitable
short-horizon structure beyond the naive baseline, using price-derived
features alone. The two results that looked like possible exceptions
(GBP/USD's ARIMA(2,0,2), USD/JPY's elevated directional accuracy) were
each investigated and traced to a specific, understood mechanism rather
than real predictive skill - which is why this is reported as a
strengthened finding, not a weaker one for having looked closer.
Full reasoning: `DECISIONS_AND_ISSUES_LOG.md` #21-#27.

## Limitations & Honest Findings

- **FX is close to a random walk, and it holds across three pairs and
  three model families.** ARIMA (linear, data-driven order search per
  pair), XGBoost (nonlinear, 32-configuration regularized grid search
  per pair), and LSTM (deep sequential, deliberately small and
  regularized, same architecture per pair) all found no exploitable
  structure in EUR/USD, GBP/USD, or USD/JPY daily returns and could not
  beat the naive "no change" baseline on any of the 18 model x pair x
  period combinations. This project is explicitly designed to report
  this honestly rather than keep tuning until a better-looking number
  appears; see `DECISIONS_AND_ISSUES_LOG.md` #12, #16, #20, and #27 for
  the full reasoning on why this is a legitimate finding, not a failure.
- **Directional accuracy is not a safe metric on its own for a trending
  series with imbalanced up/down day counts** - USD/JPY's ARIMA/LSTM
  scored above 50% on test purely because their arbitrary constant bias
  happened to align with a real, sustained trend, while XGBoost's
  opposite bias scored below 50% via the same mechanism. Always cross-
  check directional accuracy against RMSE before trusting it; see
  `DECISIONS_AND_ISSUES_LOG.md` #26.
- **GBP/USD and USD/JPY are ECB-implied cross rates** (triangulated via
  EUR/USD and EUR/GBP or EUR/JPY), not independently quoted the way
  EUR/USD is - standard methodology, verified to match Frankfurter's own
  values, but a real provenance distinction worth knowing; see
  `DECISIONS_AND_ISSUES_LOG.md` #21.
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
```

Every script below takes `--prefix` (default `eurusd`; also `gbpusd`,
`usdjpy`) and/or `--base`/`--quote`, so the full pipeline reruns
identically for any of the three pairs:

```bash
# Pull daily history (defaults to 1999-01-04 -> today)
python src/data_pull.py --base EUR --quote USD      # -> eurusd_daily.csv
python src/data_pull.py --base GBP --quote USD      # -> gbpusd_daily.csv
python src/data_pull.py --base USD --quote JPY      # -> usdjpy_daily.csv

# Exploratory data quality checks + plot (repeat --prefix for each pair)
python notebooks/01_initial_exploration.py --prefix eurusd --label "EUR/USD"

# Log returns, ADF stationarity test, return plot
python notebooks/02_target_and_stationarity.py --prefix eurusd --label "EUR/USD"

# Features, leakage check, train/val/test split
python src/feature_engineering.py --input data/processed/eurusd_daily.csv --prefix eurusd

# ACF/PACF inspection to inform ARIMA order selection
python notebooks/03_acf_pacf_check.py --prefix eurusd

# Naive baseline, ARIMA (order search + walk-forward), SARIMA seasonal check
python src/models/naive_baseline.py --prefix eurusd
python src/models/arima_sarima.py --prefix eurusd
python src/models/sarima_check.py --prefix eurusd

# XGBoost (hyperparameter grid search + feature importance)
python src/models/xgboost_model.py --prefix eurusd

# LSTM (sequence construction + leakage check + training)
python src/models/lstm_model.py --prefix eurusd

# Once all three pairs have been run through the steps above:
python src/cross_pair_comparison.py
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
│   ├── cross_pair_comparison.py  # assembles the unified 3-pair comparison table
│   ├── evaluation/
│   │   └── evaluate.py           # shared scoring module, used by every model
│   └── models/
│       ├── naive_baseline.py     # predict r_t = 0
│       ├── arima_sarima.py       # ARIMA order search + walk-forward evaluation
│       ├── sarima_check.py       # weekly-seasonal AIC pre-check
│       ├── xgboost_model.py      # hyperparameter grid search + feature importance
│       └── lstm_model.py         # sequence construction, leakage check, PyTorch LSTM
├── notebooks/
│   ├── 01_initial_exploration.py       # gap/flat-line/jump checks + plot
│   ├── 02_target_and_stationarity.py   # log return, ADF test, return plot
│   └── 03_acf_pacf_check.py            # ACF/PACF inspection for ARIMA order
└── figures/
    ├── {prefix}_1999_2026.png          # raw price plot, per pair
    ├── {prefix}_log_returns.png        # log return plot, per pair
    ├── {prefix}_acf_pacf.png           # ACF/PACF plot, per pair
    └── eurusd_lstm_loss_curve.png, gbpusd_lstm_loss_curve.png, usdjpy_lstm_loss_curve.png
```

All model scripts and notebooks above are the same file reused across
all three pairs (`--prefix eurusd|gbpusd|usdjpy`) - nothing is
duplicated per pair. `data/processed/` holds each pair's own
`{prefix}_*` outputs (splits, scaler, grid searches, model artifacts,
results) plus the final `cross_pair_comparison.csv`.
