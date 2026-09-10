"""
First-pass exploratory check on the raw EUR/USD pull: plot the series,
scan for gaps/flat-lines/implausible jumps, and report findings plainly
rather than silently "fixing" anything. Findings feed into
DECISIONS_AND_ISSUES_LOG.md, not into this script.

Run as a plain script (not committed as .ipynb output) so the checks are
easy to re-run and diff.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "eurusd_daily.csv"
FIG_PATH = REPO_ROOT / "figures" / "eurusd_1999_2026.png"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

print(f"Rows: {len(df)}")
print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"Close range: {df['close'].min()} -> {df['close'].max()}")
print(f"Any nulls: {df['close'].isnull().sum()}")
print(f"Any duplicate dates: {df['date'].duplicated().sum()}")

# --- Gap check: how many calendar days between consecutive observations ---
df = df.sort_values("date").reset_index(drop=True)
df["gap_days"] = df["date"].diff().dt.days

gap_counts = df["gap_days"].value_counts().sort_index()
print("\nGap sizes between consecutive observations (calendar days):")
print(gap_counts)

# Gaps > 4 days are worth a closer look (a normal weekend gap is 3 days,
# Fri -> Mon; a public holiday adds at most 1-2 more).
long_gaps = df[df["gap_days"] > 4][["date", "gap_days"]]
print(f"\nGaps longer than 4 days ({len(long_gaps)} found):")
print(long_gaps.to_string(index=False))

# --- Flat-line check: consecutive identical closes (data outages tend to
# repeat the last known value rather than truly not trading) ---
df["is_repeat"] = df["close"].diff() == 0
repeat_runs = df["is_repeat"].sum()
print(f"\nConsecutive identical closes (possible stale/flat data): {repeat_runs}")
if repeat_runs > 0:
    print(df.loc[df["is_repeat"], ["date", "close"]].to_string(index=False))

# --- Implausible jump check: single-day % change beyond +/-5% ---
df["pct_change"] = df["close"].pct_change() * 100
big_moves = df[df["pct_change"].abs() > 5][["date", "close", "pct_change"]]
print(f"\nSingle-day moves beyond +/-5% ({len(big_moves)} found):")
print(big_moves.to_string(index=False))

# --- Plot ---
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["date"], df["close"], linewidth=0.8)
ax.set_title("EUR/USD Daily Reference Rate (ECB via Frankfurter API)")
ax.set_xlabel("Date")
ax.set_ylabel("USD per EUR")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=150)
print(f"\nPlot saved to {FIG_PATH}")
