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

## Template for future entries (keep using this format going forward)

## N. [Short description of what happened]

**What happened:**

**Why it happened:**

**What we did:**

**Why not the alternative:**

**Concept tie-in:**
