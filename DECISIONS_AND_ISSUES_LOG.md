# Decisions & Issues Log

Running log of every real issue hit while building this project, why it happened,
what we did about it, and why we didn't take the alternative path. This is the
file to study from later - it captures the actual reasoning, not just the code.

Format for each entry:
- **What happened**
- **Why it happened**
- **What we did**
- **Why not the alternative**
- **Concept this ties to** (for review later)

---

## 1. Data source selection: exchangerate.host rejected (now requires an API key + credit card)

**What happened:** exchangerate.host, one of the three candidates specified
for this project, no longer works keyless. Every endpoint (`/latest`,
`/timeframe`) returned `{"success": false, "error": {"code": 101,
"type": "missing_access_key", ...}}` when tested live on 2026-09-10.

**Why it happened:** exchangerate.host was acquired by apilayer and folded
into their "Exchange Rates Data API" product. Its marketplace pricing page
(apilayer.com/marketplace/exchangerates_data-api) was checked directly and
displays **"Credit Card Required"** on the plan tier. This violates the
hard constraint carried over from every other project in this body of
work: no source that asks for a credit card at any tier, even a nominally
free one, is acceptable.

**What we did:** Rejected exchangerate.host outright, without attempting
signup, per the hard constraint (reject on the CC requirement alone,
don't evaluate whether the free quota would otherwise be enough).

**Why not the alternative:** Could have signed up anyway on the theory
that "we'd never actually be charged since we'd stay in the free quota" -
explicitly rejected, since the constraint is about not handing over card
details at all, not about staying under a spending limit. This mirrors
the same reasoning applied in prior projects in this body of work.

**Concept tie-in:** N/A - infra/vendor-terms issue, not a modeling concept.

---

## 2. Data source selection: Frankfurter API chosen over direct ECB SDW pull

**What happened:** Both remaining candidates - the Frankfurter API
(api.frankfurter.dev) and a direct pull from the ECB's Statistical Data
Warehouse (data-api.ecb.europa.eu) - were live-tested and both work with
no key and no signup. Cross-checked the same dates (2024-01-02 through
2024-01-10) against both sources and the EUR/USD values were **byte-for-
byte identical** (e.g. 2024-01-03: 1.0919 from both). This makes sense:
Frankfurter is an open-source wrapper (github.com/lineofflight/frankfurter)
that republishes the ECB's own daily reference rates, not an independent
data source.

**Why it happened:** Given identical underlying data, the deciding factor
was API ergonomics, not data quality. ECB SDW's native format is SDMX
(returns as CSV with ~29 columns of SDMX metadata per row, or XML) -
functional but verbose and more brittle to parse. Frankfurter returns
clean JSON keyed by date, and its range-query syntax
(`/v1/{start}..{end}?from=EUR&to=USD`) pulled the entire 1999-01-04 to
2026-09-10 history (7,090 daily rows) in a single request with no
pagination, no observed rate limit, and no chunking logic needed.

**What we did:** Chose **Frankfurter API v1** as the primary/sole data
source for this project. Documented in `src/data_pull.py` and `README.md`
that the ultimate source of truth is the ECB's daily reference rate
(published ~16:00 CET each TARGET2 business day), with Frankfurter as the
access layer.

**Why not the alternative:** Could have pulled directly from ECB SDW to
remove a third-party intermediary from the dependency chain (a legitimate
concern - Frankfurter could theoretically disappear or change terms).
Decided against it because (a) the values are verified identical, so
there's no accuracy trade-off, (b) Frankfurter's own project is open
source and self-hostable if it ever disappeared, and (c) the JSON API is
materially simpler to maintain across three currency pairs
(EUR/USD now, GBP/USD and USD/JPY later) than hand-parsing SDMX. If
Frankfurter ever becomes unavailable, `data_pull.py`'s `fetch_range()`
function is the only piece that would need to change to point at ECB SDW
directly - the cleaning/validation logic downstream is source-agnostic.

**Concept tie-in:** Evaluating a data source on more than "does it have
a free tier" - checking whether a convenience wrapper actually changes
the underlying data (it didn't, here) versus just changing how you access
it.

---

## 3. Local curl's bundled CA certificate store is stale (Anaconda-specific, not an ECB issue)

**What happened:** Testing the direct ECB SDW endpoint with the system
`curl` command initially failed with `SSL certificate problem: unable to
get local issuer certificate` (curl exit code 60), which could easily be
misread as "ECB's SDW endpoint is unreachable/broken."

**Why it happened:** This machine's Anaconda installation ships its own
`curl` on PATH, pointed at `/Users/ilakia/opt/anaconda3/ssl/cacert.pem` -
an outdated CA bundle that doesn't trust ECB's current certificate chain.
It's a local environment quirk, not a real problem with the ECB service.

**What we did:** Re-ran the same request with `/usr/bin/curl` (macOS's
system curl, with an up-to-date cert store) and it succeeded immediately
(HTTP 200, correct data). This was purely a verification-step issue during
source evaluation - Python's `requests` library (used in the actual
`data_pull.py` script, via `venv`'s own `certifi` package) was
never affected and pulled the full history without any TLS issue.

**Why not the alternative:** Could have chased down and updated Anaconda's
bundled cert file - unnecessary, since the actual project code path
(Python + requests + venv) was already unaffected; this only mattered for
a one-off manual verification command.

**Concept tie-in:** Not a modeling concept - a reminder that a tool-level
error (stale CA bundle in one specific curl binary) can look identical to
a service-level outage from the outside; worth checking whether the
failure is scoped to one tool/environment before concluding a remote
service is down.

---

## 4. First-pass data quality check on EUR/USD (1999-01-04 to 2026-09-10)

**What happened:** Ran gap, flat-line, and implausible-jump checks on the
full pulled series (7,090 daily rows) before treating it as clean.

**Findings, all judged benign (no fix applied):**
1. **Gaps between consecutive observations:** 1 day (weekday-to-weekday,
   5,614 times), 3 days (Friday-to-Monday weekend, 1,393 times), and a
   long tail of 2/4/5-day gaps (105 times combined). Every gap longer than
   3 days lines up with a known TARGET2 (Eurosystem settlement system)
   closure - Christmas/New Year and the Good Friday/Easter Monday pair
   each year - and none of the 5-day gaps are unexplained. This is
   expected market-closure behavior, not missing data, and is intentionally
   **not** imputed or forward-filled - see repo README/data docs for the
   reasoning on why weekend/holiday gaps are left as true gaps for this
   project's models to handle explicitly, rather than backfilled.
2. **55 instances of a day's close being byte-identical to the prior
   day's close** (e.g. 1999-02-03: 1.1337, matching 1999-02-02). Checked
   whether these cluster into suspicious runs (a data outage repeating a
   stale value) - they don't; each is an isolated single day scattered
   across 26 years (~0.8% of all rows), consistent with a genuinely quiet
   trading day landing on the same 4-decimal rounding, not a data quality
   defect. Left as-is.
3. **Zero single-day moves beyond +/-5%** across the entire 26-year
   history, including through the 2008 financial crisis and 2020 COVID
   shock - consistent with known EUR/USD behavior (it is a highly liquid
   pair; even crisis-period daily reference-rate moves rarely exceed
   2-3%). No implausible spikes found.
4. **Visual check:** plotted the full series
   (`figures/eurusd_1999_2026.png`) and the trajectory matches well-known
   EUR/USD history - dollar strength/EUR weakness through 2000-2001 (low
   ~0.8252 in Oct 2000), EUR rise to a pre-crisis high (~1.60 in 2008),
   the 2008 crash volatility, the 2010-2015 Eurozone debt crisis decline,
   a stable 1.05-1.25 range 2015-2021, a 2022 dip toward parity, and
   recovery to ~1.15-1.17 through 2026. No flat-lined stretches, no
   axis/unit discontinuities.

**What we did:** No cleaning beyond deduplication and date-format
normalization was necessary or applied. Documented all four checks here
rather than silently assuming the data was clean because nothing crashed.

**Why not the alternative:** Could have skipped writing this up since
"nothing was actually wrong" - explicitly against this project's own
instructions (and general good practice): an unexamined "the data looks
fine" is exactly the kind of claim that should be backed by the actual
checks run, not just visual impression.

**Concept tie-in:** Data validation before modeling - distinguishing
"true gap due to market structure" (weekends/holidays, expected, don't
impute) from "missing data" (a real problem, would need imputation or
flagging). This distinction will matter again when building lag/rolling
features later, where naive `.shift()`/`.rolling()` calls need to operate
on trading days, not calendar days.

---

## 5. Target variable: log return chosen over raw price, confirmed by ADF test (2026-09-10)

**What happened:** Before any model-fitting begins, decided what the
models actually predict. Ran an Augmented Dickey-Fuller (ADF) test on
both the raw EUR/USD close series and the daily log return
`r_t = ln(P_t / P_{t-1})` to check which one is stationary.

**Why it happened:** ARIMA (and time-series modeling generally) assumes a
stationary process — a series whose statistical properties (mean,
variance) don't drift over time. A 26-year FX price level is exactly the
kind of series that violates this: it trended from ~0.83 to ~1.60 and
back to ~1.16 over the sample, with no fixed long-run mean to revert to.
Modeling the raw price directly would break ARIMA's assumptions and also
make "did the model add value" ambiguous — a model that just predicts
"tomorrow ≈ today's price" looks deceptively accurate on a price chart
even if it has learned nothing, because 99%+ of the price level is
carried over day to day regardless of the model.

**What we did:** Ran `notebooks/02_target_and_stationarity.py`. Results:
- **Raw close price:** ADF statistic -1.93, p-value 0.320 → fails to
  reject the unit-root null hypothesis → **non-stationary**, as expected.
- **Daily log return:** ADF statistic -24.23, p-value effectively 0
  (< 1e-6) → strongly rejects the unit-root null → **stationary**.

Adopted **log return as the sole modeling target** going forward. The
raw close series is retained throughout (kept in every processed CSV) so
that return predictions can be converted back to price levels for plots
and so directional accuracy (did we get the sign right, not just the
magnitude) can be computed later. This also gives a clean, theoretically
grounded naive baseline for every model to beat: **predict r_t = 0** — the
plain random-walk hypothesis, which for a liquid, efficient pair like
EUR/USD is a genuinely tough baseline, not a strawman.

Log return summary stats (full series, 7,089 obs after the first
undefined row): mean ≈ -0.000002 (essentially zero, as expected — no
long-run drift), std ≈ 0.00581, min -0.0474, max +0.0420. The near-zero
mean is itself supporting evidence for the random-walk framing.

**Why not the alternative:** Could have modeled raw price directly and
relied on ARIMA's built-in differencing (the "I" in ARIMA/SARIMA) to
handle non-stationarity internally, which is a legitimate approach for
that one model family. Didn't take that path because the feature set and
target need to be shared/comparable across ARIMA, XGBoost, and LSTM —
XGBoost and LSTM have no equivalent built-in differencing step, so
picking a target that's stationary before it reaches *any* model keeps
the comparison fair and avoids a situation where ARIMA is implicitly
solving an easier version of the problem than the other two.

**Concept tie-in:** Stationarity and the ADF test; why comparing models
fairly requires them to see the same target definition, not just the
same raw data; random walk hypothesis as a legitimate (not naive-in-the-
pejorative-sense) FX baseline.

---

## 6. Feature engineering: lag depth, rolling windows, and the leakage check

**What happened:** Built the shared feature table
(`src/feature_engineering.py`) used by the XGBoost and LSTM models
(ARIMA/SARIMA will work primarily off the return series itself plus its
own order selection, but can pull from the same table if useful).

**Feature list:**
- `lag_1` .. `lag_5` — the previous 5 trading days' log returns
  (r_{t-1} .. r_{t-5}). Chose exactly 5 (one full trading week, Mon-Fri)
  rather than more: FX return autocorrelation is well known to be very
  weak beyond a few days for a liquid major pair (consistent with the
  ADF result above — this is close to a random walk), so extending the
  lag window further mostly adds dimensionality without adding real
  signal, and risks the XGBoost/LSTM models overfitting to lag noise.
- `roll_mean_5`, `roll_std_5` — rolling mean/std of returns over the
  prior 5 trading days (1 week).
- `roll_mean_20`, `roll_std_20` — same, over the prior 20 trading days
  (~1 trading month, standard financial-industry convention).
  `roll_std_5`/`roll_std_20` are explicitly a **realized-volatility
  proxy**, not just a generic statistical feature — their inclusion is
  motivated directly by the volatility clustering visible in the log
  return plot (`figures/eurusd_log_returns.png`): the 2008-2009 and 2022
  windows are visibly choppier than 2015-2019. Whether the models
  actually find these features useful (vs. day-of-week, below) is a
  question for the feature-importance analysis at the training stage,
  not decided here.
- `dow_mon` .. `dow_fri` — one-hot encoded day of week. Only 5 levels
  ever appear (the dataset has no weekend rows), so a plain one-hot
  encoding was used rather than a cyclical (sin/cos) encoding, which is
  more suited to a full 7-day or 12-month cycle. Included on the
  strength of the well-documented day-of-week FX liquidity pattern
  (thinner Monday/Friday activity); kept as a candidate feature for the
  model-training stage to accept or reject via importance analysis,
  deliberately not pre-judged here either way.

**Leakage check performed:** Every backward-looking feature is computed
by calling `.shift(1)` on the log-return series **before** any
`.rolling(window)` call, e.g. `roll_mean_5` = `log_return.shift(1)
.rolling(5).mean()`. This guarantees the window for row t only ever
touches r_{t-1} .. r_{t-N}, never r_t. `feature_engineering.py` also
includes an explicit runtime assertion
(`assert_no_lookahead_leakage`) that independently recomputes `lag_1`
and `roll_mean_5` from scratch and checks they match the feature table
byte-for-byte before any split/save happens — this isn't just a design
claim, it's checked every time the script runs. Day-of-week needed no
shift, since it's a calendar fact known in advance of the day's close,
not a statistic derived from returns.

**Why not the alternative:** Could have computed rolling stats per-split
(fit each window fresh within train, then within val, then within test)
to feel extra-safe about leakage — rejected, because doing so would
artificially reintroduce a NaN warm-up period at the start of val and
test (there'd be no "prior 20 days" available for the first 20 rows of
val), which is a worse and unnecessary problem. Computing on the full
continuous series first and shifting before rolling is the standard,
correct way to avoid both leakage and this warm-up artifact
simultaneously — verified above, not just assumed.

**Concept tie-in:** Look-ahead leakage in time-series feature
engineering — the most common way these projects quietly and invisibly
inflate performance; the specific `.shift(1)`-before-`.rolling()`
pattern as the standard fix; why per-split feature computation is not
actually the safer option it might first appear to be.

---

## 7. Train / validation / test split — chronological, year-boundary cutoffs

**What happened:** Split the 7,069-row feature table (after dropping 21
warm-up rows that lack a full 20-day rolling window — the very start of
1999 only) into three chronological, non-overlapping blocks:

| Split | Date range | Rows | % of data |
|---|---|---|---|
| Train | 1999-02-02 → 2019-12-31 | 5,354 | 75.7% |
| Validation | 2020-01-02 → 2023-12-29 | 1,027 | 14.5% |
| Test | 2024-01-02 → 2026-09-10 | 688 | 9.7% |

**Why it happened:** A random shuffle before splitting would let the
model train on data from *after* some of its validation/test points in
time — a direct violation of causality for a forecasting problem, and a
classic leakage source. Splitting had to be chronological.

**What we did:** Chose year-boundary cutoffs (Dec 31 / Jan 1) rather than
an exact 70/15/15 row split, favoring clean, interview-legible dates over
precise percentages — "train through 2019, validate on 2020-2023, test
on 2024-present" is easy to state and defend. This also had a useful
side effect: the **validation** window (2020-2023) happens to contain
two genuinely distinct volatility regimes — the COVID-19 crash (2020) and
the 2022 inflation/rate-hike shock — which is a more informative stretch
for hyperparameter tuning than a quieter period would be. The **test**
window (2024 to present, ~2.7 years) is held out entirely untouched,
giving a clean "how would this have performed recently" story rather
than a test period contaminated by tuning decisions.

Feature scaling (`StandardScaler`) was fit **only on the train split**
and then applied (transform, not re-fit) to validation and test — the
same shift(1)-style leakage discipline applied to preprocessing, not
just feature construction. The fitted scaler is persisted to
`data/processed/eurusd_feature_scaler.joblib` so the exact same
train-derived mean/std is reused at model-training time rather than
risking a silent re-fit later. Both raw and scaled versions of every
feature are saved in each split CSV (`*_scaled` suffix) — XGBoost doesn't
need scaling (tree splits are scale-invariant) and can use the raw
columns; LSTM will use the scaled ones.

**Why not the alternative:** Could have used a stricter 70/15/15 split by
row count, which would land on odd mid-year dates — rejected in favor of
year boundaries specifically for documentation/interview clarity, since
the ~1-2 percentage point difference this creates (75.7/14.5/9.7 vs. an
exact 70/15/15) doesn't meaningfully change what each split can teach the
model, but clean calendar-year cutoffs are much easier for a future
reader (or an interviewer) to reason about and verify independently.

**Concept tie-in:** Chronological (walk-forward-style) splitting for time
series vs. random splitting; why "which volatility regimes land in which
split" is itself a modeling decision worth making deliberately, not an
incidental side effect of wherever a percentage cutoff happens to fall;
fit-on-train-only discipline extended to scalers, not just to features.

---

## Template for future entries (keep using this format going forward)

## N. [Short description of what happened]

**What happened:**

**Why it happened:**

**What we did:**

**Why not the alternative:**

**Concept tie-in:**
