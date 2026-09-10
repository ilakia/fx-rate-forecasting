"""
First-pass exploratory check on a raw FX pull: plot the series, scan for
gaps/flat-lines/implausible jumps, and report findings plainly rather
than silently "fixing" anything. Findings feed into
DECISIONS_AND_ISSUES_LOG.md, not into this script.

Originally written for EUR/USD only (Stage 1); parameterized in Stage 6
so the same checks reapply unchanged to GBP/USD and USD/JPY.

Run as a plain script (not committed as .ipynb output) so the checks are
easy to re-run and diff.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="eurusd", help="e.g. eurusd, gbpusd, usdjpy")
    parser.add_argument("--label", default="EUR/USD", help="e.g. 'EUR/USD'")
    args = parser.parse_args()

    data_path = REPO_ROOT / "data" / "processed" / f"{args.prefix}_daily.csv"
    fig_path = REPO_ROOT / "figures" / f"{args.prefix}_1999_2026.png"

    df = pd.read_csv(data_path, parse_dates=["date"])

    print(f"=== {args.label} ({args.prefix}) ===")
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
    if repeat_runs > 0 and repeat_runs <= 60:
        print(df.loc[df["is_repeat"], ["date", "close"]].to_string(index=False))
    elif repeat_runs > 60:
        print(f"(more than 60 — showing first 20)")
        print(df.loc[df["is_repeat"], ["date", "close"]].head(20).to_string(index=False))

    # --- Implausible jump check: single-day % change beyond +/-5% ---
    df["pct_change"] = df["close"].pct_change() * 100
    big_moves = df[df["pct_change"].abs() > 5][["date", "close", "pct_change"]]
    print(f"\nSingle-day moves beyond +/-5% ({len(big_moves)} found):")
    print(big_moves.to_string(index=False))

    # --- Plot ---
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df["close"], linewidth=0.8)
    ax.set_title(f"{args.label} Daily Reference Rate (ECB via Frankfurter API)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{args.label}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"\nPlot saved to {fig_path}")


if __name__ == "__main__":
    main()
