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

## 8. pmdarima fails to import (numpy 2.0.2 ABI incompatibility) — fell back to manual order selection

**What happened:** `pip install pmdarima` succeeded, but `import pmdarima`
crashed immediately with `ValueError: numpy.dtype size changed, may
indicate binary incompatibility. Expected 96 from C header, got 88 from
PyObject`. Attempting to force a from-source rebuild
(`--no-binary pmdarima`) failed differently, with a `ModuleNotFoundError:
No module named 'distutils.msvccompiler'` coming from inside `numpy
.distutils` during pmdarima's own build script.

**Why it happened:** The venv installed numpy 2.0.2 (current stable), but
pmdarima's published wheels were compiled against an older numpy ABI —
numpy's C struct layouts changed between major versions, so a wheel built
for old numpy segfaults/errors against new numpy at import time. The
from-source fallback then hit a second, unrelated problem: pmdarima's
build script still depends on `numpy.distutils`, which numpy itself
deprecated and effectively removed, and which internally references a
Windows-only compiler module that was never going to work on this Mac
regardless of numpy version. Both routes are dead ends without either
downgrading numpy (risking compatibility issues elsewhere in the
project's dependency chain) or pmdarima shipping a numpy-2.x-compatible
release.

**What we did:** Uninstalled pmdarima and used the documented alternative
instead: manual ACF/PACF inspection (`notebooks/03_acf_pacf_check.py`)
followed by a small grid search over ARIMA(p,d,q) scored by AIC/BIC,
fit on the training split only (`src/models/arima_sarima.py`).

**Why not the alternative:** Could have pinned numpy to an older version
compatible with pmdarima's wheels — rejected, since every other piece of
this project (feature engineering, scaling, the evaluation module) was
already built and tested against numpy 2.0.2, and downgrading it this
late risks silently changing behavior elsewhere (e.g. `np.sign`,
floating-point rounding) for a single convenience library. The manual
ACF/PACF + AIC/BIC route is a legitimate, standard, and arguably more
transparent method anyway — it doesn't hide the order-selection logic
inside a black-box stepwise search.

**Concept tie-in:** Not a modeling concept — a reminder that a compiled
Python package can be functionally broken by a major dependency version
bump even when `pip install` reports success; the failure only surfaces
at import/runtime, not at install time.

---

## 9. Shared evaluation module: price-reconstruction convention and directional-accuracy tie policy

**What happened:** Before building any model, wrote `src/evaluation/evaluate.py`
so every model (naive, ARIMA now; XGBoost/LSTM later) is scored by
identical logic — two conventions had to be picked explicitly rather than
left implicit.

**Decision 1 — price reconstruction uses the TRUE previous price, never a
chained forecast.** Predicted price at day t is computed as `P_hat_t =
P_{t-1}(true) * exp(r_hat_t)`, not `P_hat_{t-1}(predicted) * exp(r_hat_t)`.

**Why:** Chaining predictions (each day's predicted price feeding the
next day's reconstruction) lets small early-period errors compound
multiplicatively over a long holdout period, so late-period price-level
error would mostly reflect accumulated drift rather than that specific
day's forecast quality — this would make the price-level RMSE/MAE
numbers much harder to interpret and would unfairly penalize (or by luck,
flatter) a model based on holdout length rather than actual skill. Using
the true previous price at every step isolates each day's forecast in
the same way the walk-forward evaluation methodology already does for
returns, keeping the two metrics (return-scale and price-scale) telling
a consistent story.

**Decision 2 — directional-accuracy tie policy: exclude, don't force a
verdict.** A day is excluded from the directional-accuracy denominator if
EITHER the true return or the predicted return is exactly 0.0.

**Why:** A return of exactly 0.0 (which genuinely occurs in this data —
55 such days, see entry #4) has no defined sign to be "right" or "wrong"
about, and neither does a model that predicts exactly 0.0. Forcing a
tie-breaking convention either way (e.g. "0 counts as positive") would
inject an arbitrary rule into the metric that doesn't reflect anything
the model actually did. Excluding both from the denominator is applied
identically to every model, so the metric stays comparable across models
even though the number of excluded days differs (see naive baseline
below — 100% of its days are ties by construction).

**Why not the alternative:** Could have counted the naive baseline's
constant 0.0 prediction as a directional "miss" every day (a common
simplistic convention) — rejected, since that would make the naive
baseline's directional accuracy a deterministic 0%, which looks like a
strong result for any other model to beat by default, when in fact the
naive baseline is making no directional claim at all and shouldn't be
scored on a metric it's structurally incapable of engaging with.

**Concept tie-in:** Designing an evaluation contract before any model
exists, specifically so no single model's quirks (e.g. naive's
degenerate direction) can silently bias the shared metric definition.

---

## 10. ARIMA order selection converges to (0,0,0) — no exploitable autocorrelation found

**What happened:** ACF/PACF inspection of the training log-return series
(`notebooks/03_acf_pacf_check.py`) showed essentially no lags outside the
approximate 95% significance band across 30 lags — visually
indistinguishable from white noise. A follow-up AIC/BIC grid search over
ARIMA(p,d,q) for p,q in {0,1,2} and d in {0,1} (d=1 included specifically
to double-check that over-differencing an already-stationary series
doesn't spuriously score better — it didn't: the best d=1 candidate's AIC
was clearly worse than every d=0 candidate) selected **ARIMA(0,0,0)** —
i.e., a plain constant-mean model with no autoregressive or moving-average
terms at all — as the best-fitting order by AIC.

**Why it happened:** This is the expected outcome given the ADF test
(#5) and the near-zero mean log return (#5's summary stats) — EUR/USD
daily returns behave close to white noise around a mean indistinguishable
from zero, so there's no linear autocorrelation structure for an ARIMA
model to exploit. This is consistent with, not contradictory to, the
random-walk framing this whole project is built around.

**What we did:** Accepted ARIMA(0,0,0) as the selected order rather than
forcing a more complex model because it "should" have more structure.
Evaluated it via one-step-ahead walk-forward forecasting across the
concatenated validation+test period (1,715 days total), where at each
step the model forecasts one day ahead, then is extended with the true
realized value. Refitting the full model at every single step would be
correct but expensive across 1,715 steps; instead the model refits its
parameters every 20 steps (~1 trading month) and simply extends its
state (no re-estimation) in between — this took ~75 seconds end to end,
against an estimated several-times-longer runtime for refit-every-step.
Order selection itself never touched validation or test data — only the
walk-forward evaluation loop sees them, one day at a time, in strict
chronological order.

**Why not the alternative:** Could have refit at every single step for
maximum accuracy-per-step — decided the 20-step cadence is a reasonable
tradeoff given how little the fitted parameters move step-to-step for a
near-constant model like this one; a future revisit with more compute
budget could re-run at refit_every=1 to confirm the results don't
meaningfully change, but given ARIMA(0,0,0) has essentially one
parameter (the mean), the risk of this cadence choice mattering is low.

**Concept tie-in:** AIC/BIC-based order selection; walk-forward
(rolling-origin) evaluation for time series, as distinct from either a
single train/test fit-once-predict-all-at-once evaluation or a
multi-step compounded forecast; refit cadence as a genuine
compute-vs-freshness tradeoff, not a shortcut taken without justification.

---

## 11. SARIMA (weekly seasonal terms) rejected on AIC evidence, not run through full walk-forward

**What happened:** Given the known FX day-of-week liquidity pattern
observed during feature engineering (#6), tested whether adding weekly
(m=5, one trading week) seasonal terms on top of the winning ARIMA(0,0,0)
improves in-sample fit. Grid-searched SARIMA(0,0,0)x(P,0,Q,5) for P,Q in
{0,1} against the plain ARIMA(0,0,0) baseline, all fit on train only.

**Result:** Best seasonal candidate's AIC was *worse* than plain
ARIMA(0,0,0) (AIC improvement of -1.6, i.e. actually negative — adding
seasonal terms made the fit slightly worse once the extra parameters were
penalized), well under the conventional "improvement of at least 2" rule
of thumb for AIC to call a more complex model meaningfully better.

**Why it happened:** This is consistent with, not contradictory to, the
ACF/PACF check (#10 concept) — the day-of-week *liquidity* pattern
(thinner Monday/Friday trading volume, a real and well-documented FX
phenomenon) doesn't necessarily translate into a *return* pattern at the
same weekly lag. Liquidity and directional predictability are different
things; this data shows evidence for neither weekly return autocorrelation
nor a seasonal model improving on the non-seasonal one.

**What we did:** Rejected SARIMA based on this in-sample AIC evidence
alone and did **not** run it through the ~75-second walk-forward
evaluation, since there was no positive in-sample signal to justify the
extra compute. Documented this as a real, reportable finding (no
exploitable weekly seasonality in EUR/USD returns) rather than silently
dropping the seasonal-terms idea without a paper trail.

**Why not the alternative:** Could have run the full walk-forward
evaluation on SARIMA anyway "just to be thorough" — decided against it,
since the in-sample AIC comparison is the appropriate gate for this
decision (a seasonal model that already fits worse in-sample, on the
very data it was estimated from, has no plausible path to outperforming
out-of-sample) and running the expensive walk-forward regardless would
have been effort spent without a decision-relevant question left to
answer.

**Concept tie-in:** Using in-sample AIC as an efficient pre-filter before
committing to expensive out-of-sample evaluation; the AIC "improvement of
2" rule of thumb; distinguishing a liquidity/volume seasonal pattern from
a return/predictability seasonal pattern — they are not the same claim.

---

## 12. Honest finding: ARIMA(0,0,0) is statistically indistinguishable from the naive baseline

**What happened:** Assembled the final val/test comparison table for
Stage 3 (`data/processed/model_comparison_stage3.csv`):

| Model | Period | RMSE (return) | MAE (return) | RMSE (price) | MAE (price) | Directional accuracy |
|---|---|---|---|---|---|---|
| Naive | val | 0.004976 | 0.003664 | 0.005412 | 0.004035 | undefined (100% ties) |
| ARIMA(0,0,0) | val | 0.004976 | 0.003664 | 0.005412 | 0.004036 | 49.2% (8 ties excluded) |
| Naive | test | 0.004212 | 0.002980 | 0.004687 | 0.003329 | undefined (100% ties) |
| ARIMA(0,0,0) | test | 0.004212 | 0.002981 | 0.004688 | 0.003329 | 51.0% (6 ties excluded) |

RMSE/MAE match to 3-4 significant figures on both val and test. ARIMA's
directional accuracy (49.2% val, 51.0% test) sits right on top of a coin
flip (50%), with no consistent edge in either direction across the two
periods.

**Why it happened:** ARIMA(0,0,0) is, mechanically, a constant equal to
the training mean log return (a tiny, near-zero number) — so its
predictions are nearly identical to naive's flat 0.0 prediction on every
single day, and the resulting RMSE/MAE are almost mathematically
guaranteed to match closely. Its directional accuracy being ~50% simply
reflects that the sign of a tiny near-zero constant relative to a
genuinely noisy return series carries no real predictive information —
it's arbitrary from the model's perspective, not evidence of skill.

**What we did:** Reported this plainly as the Stage 3 headline finding,
framed as a genuine and expected result about EUR/USD market behavior —
**a linear time-series model finds no exploitable autocorrelation
structure in daily returns, consistent with the (weak-form) efficient
market / random-walk hypothesis for a highly liquid FX pair** — rather
than as a project shortfall. No repeated re-tuning of the ARIMA order
against test-set performance was done to try to manufacture a better-
looking number; the order was selected once, against train only, before
either validation or test was scored.

**Why not the alternative:** Could have kept searching wider (p,q) ranges,
different differencing, or exotic ARIMA variants specifically until
something beat naive on test — explicitly avoided, since that would be
p-hacking against the test set (the exact anti-pattern this project's own
instructions call out), and because the ACF/PACF evidence (#10) already
gives a principled reason to expect this outcome rather than treating it
as a search that just hasn't found the right order yet. This result also
sets a clear, honest bar: XGBoost and LSTM (nonlinear models, capable of
finding structure a linear ARIMA cannot) are the more interesting test of
whether *any* exploitable signal exists in this feature set — if they
also fail to beat naive, that becomes an even stronger finding about this
specific problem, not a weaker one.

**Concept tie-in:** The random walk / efficient market hypothesis as an
empirically testable claim, not just a theoretical assumption; why
"the fancy model didn't beat the simple one" is itself a legitimate and
reportable research finding rather than something to keep tuning away;
avoiding test-set p-hacking by fixing the tuning boundary (validation
only) before ever computing a test score.

---

## 13. XGBoost evaluated with a single fit, not walk-forward — a justified difference from ARIMA, not an inconsistency

**What happened:** ARIMA (#10) was evaluated with one-step-ahead
walk-forward forecasting, refitting periodically as new true values
arrived. XGBoost (`src/models/xgboost_model.py`) is instead fit once on
train and evaluated with a plain one-shot `.predict()` on val and test.

**Why it happened:** The two models have fundamentally different
relationships to "new information." ARIMA's fitted parameters (a mean,
or AR/MA coefficients) are a compressed summary of everything the model
has seen — that summary can go stale as the process drifts, which is
exactly why walk-forward re-exposes it to fresh true values one step at
a time. XGBoost's *features*, by contrast, are already genuine
backward-looking historical values (real past returns, real past
rolling stats) computed once for every row of the feature table back at
Stage 2 — a validation or test row's features already encode "true data
known up to that point," independent of when the model itself was fit.
There is no equivalent "staleness" for a single fixed fit to correct for.

**What we did:** Documented this reasoning directly in
`xgboost_model.py`'s module docstring so a reader doesn't assume the two
models were scored on different, incomparable standards. Did not build
the optional periodic-retrain sanity-check variant (e.g. retrain every
N months and compare) — skipped deliberately, since the single-fit
result already matches naive/ARIMA almost exactly on both val and test
(see #16), leaving no drift-related question a periodic-retrain variant
would meaningfully help answer; it would cost real compute to check
something the evidence already speaks to.

**Why not the alternative:** Could have forced XGBoost through the same
walk-forward-with-refit loop "for consistency" — rejected, since that
consistency would be superficial (same procedure, not same
justification) and walk-forward exists specifically to solve a staleness
problem that doesn't apply to a feature-table-based model in the same way.

**Concept tie-in:** Evaluation methodology should match *why* a model
might need re-exposure to new data, not be copy-pasted across model
families for the appearance of uniformity.

---

## 14. Feature-importance bug caught: `get_score()` includes trees never used for prediction

**What happened:** First pass at extracting XGBoost feature importance
called `booster.get_score(importance_type="gain")` directly on the fitted
model and got a suspiciously broad importance table — 11 of 14 features
showed nonzero gain, including several day-of-week columns, even though
the model's validation performance was statistically identical to naive
(no evidence it had learned anything meaningful).

**Why it happened:** With `early_stopping_rounds=50`, XGBoost keeps
boosting for 50 rounds *past* the best-scoring round before stopping —
in this run, `best_iteration=0` but training continued to round 51
before patience was exhausted, meaning **50 extra trees were built and
included in the booster object that are never actually used at
prediction time** (`model.predict()` correctly restricts itself to
`iteration_range=(0, best_iteration+1)`, verified directly). `get_score()`,
however, aggregates gain across **every tree ever built in the booster**,
with no automatic restriction to the deployed range — so it was reporting
importance from 50 discarded trees the model doesn't actually use.

**What we did:** Verified the discrepancy directly: confirmed
`model.predict()` output matches `booster.predict(..., iteration_range=(0,1))`
exactly (i.e., only tree 0 is deployed), then re-derived feature
importance correctly via `booster.trees_to_dataframe()`, filtered to
`Tree <= best_iteration`, and summed gain only over that deployed range.
The corrected result is much narrower and more honest: only 3 of 14
features (`roll_std_20`, `lag_1`, `lag_5`) have any nonzero gain at all,
and the total gain across all three is tiny (~0.0005) — consistent with
a single shallow, barely-useful tree, not a model that found real
structure. Day-of-week features: **exactly zero gain** — they were never
split on in the deployed tree.

**Why not the alternative:** Could have reported the original, broader
`get_score()` table since it "ran without error" — would have been
actively misleading: attributing predictive importance to features
(several `dow_*` columns) that provably never influence a single
prediction the deployed model makes. This is exactly the kind of
plausible-looking-but-wrong result this project's own instructions warn
about scrutinizing before accepting.

**Concept tie-in:** Early stopping's "patience" window silently keeps
building (and, by default, reporting importance from) trees the deployed
model doesn't use — a genuinely non-obvious XGBoost API gotcha worth
knowing before trusting `get_score()` output on any early-stopped model.

---

## 15. Train-vs-validation RMSE comparison is confounded by different volatility regimes, not overfitting

**What happened:** The XGBoost model's train-set RMSE (0.006126) came out
*higher* than its validation-set RMSE (0.004976) — at a glance this looks
backwards (models are supposed to fit training data at least as well as
held-out data), which could be misread as a bug.

**Why it happened:** Checked the unconditional standard deviation of log
returns in each split independently: train (1999-2019) std = 0.006127,
val (2020-2023) std = 0.004978, test (2024-2026) std = 0.004214. The
train period's own return volatility (which includes the dot-com
unwind and the 2008 financial crisis) is genuinely higher than either
the validation or test period's volatility — confirming this is the
same volatility-clustering phenomenon already noted from the raw return
plot (#5, #6), not a modeling artifact. A model that has converged to
predicting essentially zero everywhere (see #16) will naturally show a
higher RMSE on a noisier period and a lower RMSE on a calmer one, with
zero relationship to overfitting.

**What we did:** Replaced the naive train-vs-val RMSE subtraction with
the correct comparison: each period's model RMSE against **that same
period's own naive-baseline RMSE**. Train: 0.006126 vs. naive 0.006126
(ratio 1.0000). Val: 0.004976 vs. naive 0.004976 (ratio 1.0000). The
model matches naive on both periods individually — meaning early
stopping converged to an essentially trivial prediction everywhere, not
a model that overfit train's noise and then failed to generalize (which
would show a *good* train ratio and a *bad* val ratio, not two ratios
that are both ~1.0).

**Why not the alternative:** Could have reported the raw negative
train-val gap as "no overfitting, val is even better than train!" without
investigating why — would have missed the more informative and correct
explanation (different volatility regimes) and could have looked like a
red flag to a careful reader without the explanation attached.

**Concept tie-in:** The standard "compare train performance to val
performance" overfitting heuristic implicitly assumes both periods share
similar underlying variance — when they don't (as here, given documented
volatility clustering), the comparison needs to be re-anchored to each
period's own trivial-baseline performance instead of compared directly
against each other.

---

## 16. Honest finding: XGBoost, like ARIMA, does not beat the naive baseline

**What happened:** Ran a 32-combination grid search over regularization
hyperparameters (max_depth ∈ {2,3}, min_child_weight ∈ {10,20}, subsample
∈ {0.7,0.8}, colsample_bytree ∈ {0.7,0.8}, learning_rate ∈ {0.01,0.05}),
each trained with up to 1,000 boosting rounds and early stopping
(patience=50) on validation RMSE. **Every single one of the 32
configurations independently converged to `best_iteration=0`** — i.e.,
the very first tree already minimized validation RMSE, and every
additional tree in every configuration made validation performance
worse, with validation RMSE across all 32 configs varying only in the
5th decimal place (0.004976-0.004977).

Extended the Stage 3 comparison table:

| Model | Period | RMSE (return) | MAE (return) | Directional accuracy |
|---|---|---|---|---|
| Naive | val | 0.004976 | 0.003664 | undefined |
| ARIMA(0,0,0) | val | 0.004976 | 0.003664 | 49.2% |
| XGBoost | val | 0.004976 | 0.003664 | 51.2% |
| Naive | test | 0.004212 | 0.002980 | undefined |
| ARIMA(0,0,0) | test | 0.004212 | 0.002981 | 51.0% |
| XGBoost | test | 0.004212 | 0.002980 | 51.6% |

RMSE/MAE match all three models to 3-4 significant figures on both
periods. XGBoost's directional accuracy (51.2% val, 51.6% test) is
marginally higher than ARIMA's but still well within coin-flip range —
no model here shows a directional edge that would survive normal
statistical scrutiny (no significance test was run to formally confirm
this, but a ~1-2 percentage point difference on ~700-1000 observations
is not a claim this project is making).

**Why it happened:** Consistent with the ACF/PACF finding (#10) that
train-period returns carry essentially no linear autocorrelation
structure, and now extended to nonlinear structure too: even a flexible,
regularization-constrained gradient-boosted-tree model, searched across
32 hyperparameter configurations, found nothing worth building past a
single weak tree. The one deployed tree's feature importance (corrected
per #14) attributes its entire (tiny) contribution to `roll_std_20`
(realized volatility), `lag_1`, and `lag_5` — everything else, including
all five day-of-week columns, has exactly zero importance.

**What we did:** Reported this plainly as a second, independent
confirmation of the random-walk finding from Stage 3, not as a modeling
failure. Specifically addressed the two features flagged as "candidate,
not pre-judged" in Stage 2: **day-of-week features show zero importance**
in the deployed model — the known FX day-of-week liquidity pattern does
not translate into next-day return predictability, at least not one this
model could exploit. **Rolling-volatility (`roll_std_20`) is the single
most-used feature**, though its contribution is still tiny in absolute
terms — a hint (not strong evidence, given the model doesn't actually
beat naive) that realized volatility might carry more information than
lagged returns do, consistent with volatility clustering being a real,
established FX phenomenon even when *direction* remains unpredictable.
No further hyperparameter tuning was attempted once regularization and
early stopping were in place and consistently converged to the same
null result — per this project's own instructions, a modest, well-
justified result is the goal here, not a maximally-tuned one.

**Why not the alternative:** Could have widened the search (deeper
trees, weaker regularization, more features) specifically until
something beat naive on validation — explicitly avoided, both because
it risks eventually overfitting *to validation* (a slower-motion version
of the test-set p-hacking already avoided in #12) and because 32
configurations landing on the identical null result is itself strong,
convergent evidence rather than an under-searched space.

**Concept tie-in:** A tree ensemble's early stopping mechanism, used
correctly, is itself an overfitting guard — 32/32 configurations
independently refusing to boost past round 0 is a stronger and more
convincing "no signal here" signal than a single model's result would
be; gain-based feature importance as a lens for distinguishing which
candidate features carry even marginal information, separate from
whether the model's overall predictions are useful.

---

## 17. TensorFlow crashes on import (AVX instructions unavailable under Rosetta 2) — switched to PyTorch

**What happened:** `pip install tensorflow` succeeded, but
`import tensorflow` crashed immediately with SIGABRT (exit code 134) and
the message "The TensorFlow library was compiled to use AVX
instructions, but these aren't available on your machine."

**Why it happened:** Checked `platform.machine()` inside the project's
venv and got `x86_64`, on a machine whose actual CPU is Apple Silicon
(arm64) — confirming this venv's Python (inherited from the system's
Anaconda base Python since the very first `python3 -m venv venv` back in
Stage 1) is an x86_64 build running under Rosetta 2 translation, not a
native arm64 Python. This had been silently true for every prior stage
of this project without causing a problem, because pandas/numpy/
statsmodels/xgboost's x86_64 wheels apparently don't hard-require AVX
(or at least degrade gracefully). TensorFlow's official pip wheel does
hard-require AVX and aborts immediately if it's missing — and Rosetta 2
specifically does **not** emulate AVX/AVX2 instructions (a documented
Rosetta 2 limitation), so any AVX-requiring x86_64 binary crashes
outright when run through it on Apple Silicon.

**What we did:** Rather than rebuild the entire project's venv from an
arm64-native Python (which would mean reinstalling and re-verifying every
package used since Stage 1, a large blast radius this late in the
project, for a problem that's local to one library), switched the LSTM
implementation from TensorFlow/Keras to **PyTorch** (CPU build,
`pip install torch --index-url https://download.pytorch.org/whl/cpu`),
which imported and ran without incident. This only changes which library
implements the model — the architecture, training procedure, evaluation
methodology, and results are unaffected by this choice, and PyTorch is
an equally standard, widely-used framework for exactly this kind of model.

**Why not the alternative:** Could have created a second, arm64-native
venv (e.g. from `/usr/bin/python3`, confirmed arm64-native on this
machine) just for the LSTM stage — rejected, since maintaining two
separate Python environments for one project, with different
architectures and dependency sets, adds real reproducibility risk (a
future reader following "How to Reproduce" would need to know which venv
runs which script) for no benefit once a same-venv fix (switching
libraries) was available and worked immediately.

**Concept tie-in:** Not a modeling concept — a reminder that a machine's
reported architecture (`uname -m` showing `arm64`) doesn't guarantee
every process on it runs natively; a specific Python interpreter/venv can
be running under binary translation the whole time without any visible
symptom, until a library with a stricter hardware requirement (AVX here)
surfaces it.

---

## 18. LSTM sequence construction: window choice, per-split boundaries, and the leakage check

**What happened:** Built `src/models/lstm_model.py`'s `build_sequences()`
to turn the Stage 2 feature table into (sequence, target) pairs for the
LSTM — the first model in this project to consume the feature set as an
actual temporal sequence rather than one flattened row per prediction.

**Window length: 20 trading days (~1 trading month).** Chosen to match
the "long" rolling-stat window already established in Stage 2
(`roll_std_20`/`roll_mean_20`) for the same underlying reasoning: long
enough that a sequence can span more than one short-term volatility
regime (the visible clustering in `figures/eurusd_log_returns.png` plays
out over multi-week stretches, not single days), short enough that it
doesn't meaningfully shrink the ~5,300-row training set. Reusing an
already-justified window length, rather than picking a new arbitrary
number, keeps the project's reasoning consistent across stages.

**Per-split sequence boundaries.** Unlike XGBoost's single-row features
(which could safely reach across a split boundary, since those features
were computed once on the full continuous series back in Stage 2 and
splitting happens afterward), LSTM sequences are built **independently
per split** — a validation sequence's 20-day window never reaches back
into training data, and a test sequence's window never reaches back into
validation data. This is a stricter, more conservative boundary than
strictly required (the underlying feature values are already "true past
data" regardless of split, same as for XGBoost), chosen because a
sequence model's whole premise is temporal continuity, and mixing
which-split-supplied-which-timestep into one sequence adds a layer of
bookkeeping complexity for no clear benefit here. The cost: each split's
first 20 rows become unusable as prediction targets and are dropped
(train 5,354 -> 5,334 sequences; val 1,027 -> 1,007; test 688 -> 668) —
a small, explicitly documented reduction, not a silent one.

**Leakage check.** `check_no_leakage()` independently verifies, for every
sequence: (1) the last input row's date is strictly before the target
date, (2) the target row immediately follows the last input row in the
sorted dataframe (ruling out an off-by-one or gap slipping a future row
into the window), and (3) for spot-checked sequences, the constructed
input array matches an independently re-sliced window straight from the
dataframe. This mirrors the same "recompute independently and compare"
discipline Stage 2 applied to the lag/rolling features (#6), extended to
a genuinely different data shape (3D sequences vs. flat rows).

**Why not the alternative:** Could have let sequences span split
boundaries (using the last ~20 days of train to seed val's earliest
sequences, etc.), which would preserve slightly more usable rows per
split — decided against it for the reason above (avoiding cross-split
bookkeeping complexity), and because the number of rows lost (60 total
across all three splits) is small relative to the total ~7,000-row
dataset.

**Concept tie-in:** Sequence-to-target alignment in recurrent models is
a distinct leakage surface from the lag/rolling-feature leakage already
handled in Stage 2 — same underlying principle (never let the model see
information from at-or-after the prediction point), different mechanism
to verify (index/date alignment across a 3D array, not a `.shift()` call).

---

## 19. LSTM architecture: kept deliberately small, given the null result from every prior model

**What happened:** Built a minimal architecture: a single `LSTM(hidden
_size=16)` layer, dropout (0.2) on its output, and a single `Linear(16,
1)` output layer — 2,065 trainable parameters total, trained with Adam
(lr=1e-3), batch size 32, up to 100 epochs with early stopping
(patience=10) on validation loss.

**Why it happened:** By this stage, both a linear model (ARIMA) and a
flexible nonlinear tabular model (XGBoost, searched across 32
hyperparameter configurations) had found no exploitable structure in
this feature set. Starting from a large/deep architecture (multiple
stacked LSTM layers, large hidden sizes, hundreds of thousands of
parameters) on data that has now twice shown itself to be close to
white noise would mean giving the model far more capacity to fit noise
in the training set than there is real signal to find — a bigger model
is not a more rigorous test of "is there signal here" once two other
approaches have already characterized the signal as weak-to-absent; it's
mainly a bigger overfitting risk.

**What we did:** Started small and let the training curve itself confirm
or challenge that choice, rather than assuming it. Training/validation
loss both dropped quickly and converged to a stable plateau by
~epoch 15-20 (`figures/lstm_loss_curve.png`), with early stopping firing
at epoch 56 (best epoch 46) after 10 epochs of no further validation
improvement — the two curves converge together and stay flat, with no
divergence pattern (train continuing to drop while val flattens or rises)
that would indicate the model is large enough to be overfitting. This
is itself evidence the modest architecture was an appropriate choice for
this problem, not an undersized one straining against real signal it
couldn't capture.

**Why not the alternative:** Could have tried a larger architecture "just
to see" — deliberately avoided per this project's own instructions not
to chase a better-looking number once regularization/early stopping are
reasonably in place, and because the flat, converged, non-diverging
loss curve gives no indication a larger model would find anything
different — a larger model on this same data would most likely reach the
same plateau, just with a longer and more expensive path there (and a
higher risk of finding spurious in-sample structure along the way).

**Concept tie-in:** Model capacity should be chosen relative to the
evidenced amount of learnable signal, not maximized by default — this is
the same reasoning already applied to XGBoost's regularization (#16),
now applied to neural network sizing.

---

## 20. Final EUR/USD finding: three independent model families all confirm the same null result

**What happened:** Evaluated the LSTM through the shared module and
extended the running comparison table to all four approaches:

| Model | Period | RMSE (return) | MAE (return) | Directional accuracy |
|---|---|---|---|---|
| Naive | val | 0.004976 | 0.003664 | undefined |
| ARIMA(0,0,0) | val | 0.004976 | 0.003664 | 49.2% |
| XGBoost | val | 0.004976 | 0.003664 | 51.2% |
| LSTM | val | 0.005003 | 0.003697 | 49.4% |
| Naive | test | 0.004212 | 0.002980 | undefined |
| ARIMA(0,0,0) | test | 0.004212 | 0.002981 | 51.0% |
| XGBoost | test | 0.004212 | 0.002980 | 51.6% |
| LSTM | test | 0.004282 | 0.003032 | 50.6% |

(LSTM's val/test row counts are 1,007/668 rather than 1,027/688 due to
the 20-row sequence warm-up per split, #18 — the naive/ARIMA/XGBoost rows
above are on the full split sizes, so LSTM's RMSE is not perfectly
apples-to-apples row-for-row with the other three, though the 20 dropped
rows at the start of each split are not a systematically different
period and wouldn't plausibly explain the small gap on their own.)

**Why it happened:** LSTM's RMSE is marginally *worse* than the other
three (0.005003 vs. 0.004976 on val; 0.004282 vs. 0.004212 on test) —
a small, consistent overhead rather than an improvement. This is
consistent with a model that, like XGBoost, found no real predictive
structure, but — unlike XGBoost's early-stopping-at-round-0 — still
carries some inherent estimation noise from its parameters not landing
at the exact trivial-prediction optimum, paying a tiny generalization
cost for having free parameters at all. Directional accuracy (49.4%
val, 50.6% test) again sits squarely in coin-flip range.

**What we did:** Reported this as the third and final confirmation of
the same finding already established twice (#12, #16), now covering all
three planned model families. Drafted the project's headline finding for
the README: **three independent modeling approaches — linear statistical
(ARIMA), nonlinear tabular (XGBoost, 32/32 configurations), and deep
sequential (LSTM) — all confirm that EUR/USD daily returns show no
exploitable structure beyond the naive "no change" baseline**, consistent
with weak-form market efficiency in a highly liquid FX pair. This is
reported as a real, defensible research outcome, not as a project
shortfall — see README "Key Finding" section.

**Why not the alternative:** Could have kept adjusting the LSTM
(different window length, larger hidden size, additional features)
specifically until it beat naive on validation — explicitly avoided,
for the same test/validation-integrity reasons already established in
#12 and #16, and because three independently-built, differently-shaped
models landing on the same conclusion is a *stronger* result than any
one of them individually, not a reason to keep searching for an
exception.

**Concept tie-in:** Converging evidence across structurally different
model families (linear, tree-based, recurrent) as a stronger form of
validation than repeated tuning of a single model family; a small,
consistent RMSE overhead (rather than a dramatic miss) as the expected
signature of a flexible model that found no signal but still incurs
some estimation variance, as distinct from a model that's actively wrong.

---

## Template for future entries (keep using this format going forward)

## N. [Short description of what happened]

**What happened:**

**Why it happened:**

**What we did:**

**Why not the alternative:**

**Concept tie-in:**
