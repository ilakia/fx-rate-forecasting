"""
Compute the daily log-return target from a cleaned FX close series, run
an Augmented Dickey-Fuller (ADF) test on raw price vs. log return to
back up the target-variable decision with actual evidence, and plot the
return series.

Originally written for EUR/USD only (Stage 2); parameterized in Stage 6
so stationarity is independently re-confirmed for GBP/USD and USD/JPY
rather than assumed to transfer from EUR/USD's result.

Run as a plain script (matches 01_initial_exploration.py's convention).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_adf(series: pd.Series, label: str) -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    stat, pvalue, used_lag, nobs, crit_values, _ = result
    print(f"\nADF test — {label}")
    print(f"  Test statistic: {stat:.4f}")
    print(f"  p-value:        {pvalue:.6f}")
    print(f"  # lags used:    {used_lag}")
    print(f"  # observations: {nobs}")
    print(f"  Critical values: {crit_values}")
    stationary = pvalue < 0.05
    print(f"  -> {'STATIONARY' if stationary else 'NON-STATIONARY'} at 5% significance")
    return {
        "label": label,
        "statistic": stat,
        "pvalue": pvalue,
        "used_lag": used_lag,
        "nobs": nobs,
        "critical_values": crit_values,
        "stationary_at_5pct": stationary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="eurusd", help="e.g. eurusd, gbpusd, usdjpy")
    parser.add_argument("--label", default="EUR/USD", help="e.g. 'EUR/USD'")
    args = parser.parse_args()

    data_path = REPO_ROOT / "data" / "processed" / f"{args.prefix}_daily.csv"
    fig_path = REPO_ROOT / "figures" / f"{args.prefix}_log_returns.png"

    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    print(f"=== {args.label} ({args.prefix}) ===")
    adf_price = run_adf(df["close"], f"{args.label} raw close price")
    adf_return = run_adf(df["log_return"], f"{args.label} daily log return")

    print(f"\nLog return summary stats:")
    print(df["log_return"].describe())

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df["log_return"], linewidth=0.5, color="darkred")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"{args.label} Daily Log Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("log(P_t / P_{t-1})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"\nPlot saved to {fig_path}")

    return adf_price, adf_return


if __name__ == "__main__":
    main()
