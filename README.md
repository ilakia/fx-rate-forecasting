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

## Methodology (to come)

Planned progression:
1. Feature engineering (lags, rolling stats, calendar features) - trading-day
   aware, not calendar-day aware.
2. Naive persistence baseline (mandatory reference point for everything else).
3. ARIMA/SARIMA.
4. XGBoost.
5. LSTM.

Not yet started - see "What NOT to do in this session" note: this phase
was scoped to data source selection and the initial pull only.

## Results (to come)

No models have been built yet. Nothing here is fabricated ahead of that
work.

## Limitations & Honest Findings

- **Only EUR/USD has been pulled so far.** GBP/USD and USD/JPY are planned
  but not yet built - the data pull script (`src/data_pull.py`) is
  parameterized by `--base`/`--quote` specifically so the same logic
  reapplies without rewriting.
- **FX is close to a random walk.** This project is explicitly designed to
  report honestly if none of the more sophisticated models beat a naive
  baseline - that would be a real and useful finding, not a failed project.
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
```

## Repo Structure

```
fx-rate-forecasting/
├── README.md
├── DECISIONS_AND_ISSUES_LOG.md   # full engineering/methodology log
├── requirements.txt
├── data/
│   ├── raw/                      # untouched API responses
│   └── processed/                # cleaned, deduplicated daily series
├── src/
│   └── data_pull.py              # reusable pull script, any currency pair
├── notebooks/
│   └── 01_initial_exploration.py # gap/flat-line/jump checks + plot
└── figures/
    └── eurusd_1999_2026.png
```
