"""
ACF/PACF inspection of the training log-return series, used to narrow the
ARIMA/SARIMA order search in src/models/arima_sarima.py. Also checks for
weekly (lag-5) seasonal structure, since day-of-week FX liquidity patterns
were observed at the feature-engineering stage — this notebook checks
whether that translates into actual RETURN predictability at a seasonal
lag, which is a different and stronger claim.
"""

from pathlib import Path

import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = REPO_ROOT / "data" / "processed" / "eurusd_train.csv"
FIG_PATH = REPO_ROOT / "figures" / "eurusd_acf_pacf.png"

train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
r = train["log_return"]

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(r, lags=30, ax=axes[0], title="ACF — training log returns (30 lags)")
plot_pacf(r, lags=30, ax=axes[1], method="ywm", title="PACF — training log returns (30 lags)")
fig.tight_layout()
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=150)
print(f"Saved ACF/PACF plot to {FIG_PATH}")

# Print raw ACF values at lags 1-10 and the weekly lags 5, 10, 15, 20 for
# an explicit look at whether lag-5 (one trading week) stands out.
from statsmodels.tsa.stattools import acf
acf_vals = acf(r, nlags=20, fft=True)
print("\nACF values, lags 1-20:")
for lag in range(1, 21):
    marker = "  <- weekly lag" if lag % 5 == 0 else ""
    print(f"  lag {lag:2d}: {acf_vals[lag]: .4f}{marker}")

# Rough significance band for a series of this length: +/- 1.96/sqrt(N)
import numpy as np
n = len(r.dropna())
band = 1.96 / np.sqrt(n)
print(f"\nApprox 95% significance band: +/-{band:.4f}")
print(f"Lags with |ACF| exceeding the band: "
      f"{[lag for lag in range(1, 21) if abs(acf_vals[lag]) > band]}")
