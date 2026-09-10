"""
ACF/PACF inspection of a pair's training log-return series, used to
narrow the ARIMA/SARIMA order search in src/models/arima_sarima.py. Also
checks for weekly (lag-5) seasonal structure, since day-of-week FX
liquidity patterns were observed at the feature-engineering stage — this
notebook checks whether that translates into actual RETURN predictability
at a seasonal lag, which is a different and stronger claim.

Originally written for EUR/USD only (Stage 3); parameterized in Stage 6
so the same check reapplies to GBP/USD and USD/JPY independently.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="eurusd", help="e.g. eurusd, gbpusd, usdjpy")
    args = parser.parse_args()

    train_path = REPO_ROOT / "data" / "processed" / f"{args.prefix}_train.csv"
    fig_path = REPO_ROOT / "figures" / f"{args.prefix}_acf_pacf.png"

    train = pd.read_csv(train_path, parse_dates=["date"])
    r = train["log_return"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(r, lags=30, ax=axes[0], title=f"ACF — {args.prefix} training log returns (30 lags)")
    plot_pacf(r, lags=30, ax=axes[1], method="ywm", title=f"PACF — {args.prefix} training log returns (30 lags)")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Saved ACF/PACF plot to {fig_path}")

    # Print raw ACF values at lags 1-20 and flag the weekly lags 5, 10, 15, 20.
    acf_vals = acf(r, nlags=20, fft=True)
    print("\nACF values, lags 1-20:")
    for lag in range(1, 21):
        marker = "  <- weekly lag" if lag % 5 == 0 else ""
        print(f"  lag {lag:2d}: {acf_vals[lag]: .4f}{marker}")

    # Rough significance band for a series of this length: +/- 1.96/sqrt(N)
    n = len(r.dropna())
    band = 1.96 / np.sqrt(n)
    print(f"\nApprox 95% significance band: +/-{band:.4f}")
    print(f"Lags with |ACF| exceeding the band: "
          f"{[lag for lag in range(1, 21) if abs(acf_vals[lag]) > band]}")


if __name__ == "__main__":
    main()
