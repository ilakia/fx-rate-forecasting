"""
Compute the daily log-return target from the cleaned EUR/USD close series,
run an Augmented Dickey-Fuller (ADF) test on raw price vs. log return to
back up the target-variable decision with actual evidence, and plot the
return series.

Run as a plain script (matches 01_initial_exploration.py's convention).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "eurusd_daily.csv"
FIG_PATH = REPO_ROOT / "figures" / "eurusd_log_returns.png"


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
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    adf_price = run_adf(df["close"], "raw close price")
    adf_return = run_adf(df["log_return"], "daily log return")

    print(f"\nLog return summary stats:")
    print(df["log_return"].describe())

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df["log_return"], linewidth=0.5, color="darkred")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("EUR/USD Daily Log Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("log(P_t / P_{t-1})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nPlot saved to {FIG_PATH}")

    return adf_price, adf_return


if __name__ == "__main__":
    main()
