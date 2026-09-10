"""
Secondary check: does adding weekly (m=5, one trading week) seasonal terms
to the winning ARIMA(0,0,0) improve in-sample fit on train? ACF/PACF
inspection (notebooks/03_acf_pacf_check.py) found no strong evidence of
weekly seasonal autocorrelation in returns (lag-5/15/20 ACF values were
inside or barely at the noise band), so this is run as a documented
double-check rather than a strong prior — if AIC doesn't improve, SARIMA
is rejected without spending the ~90s walk-forward cost on it.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"

BASE_ORDER = (0, 0, 0)  # winning non-seasonal order from arima_sarima.py
SEASONAL_PERIOD = 5  # one trading week


def main():
    train = pd.read_csv(PROCESSED / "eurusd_train.csv", parse_dates=["date"])
    train_r = train.set_index("date")["log_return"]

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SARIMAX(train_r, order=BASE_ORDER,
                        seasonal_order=(0, 0, 0, 0)).fit(disp=False)
        rows.append({"P": 0, "D": 0, "Q": 0, "m": 0, "aic": base.aic, "bic": base.bic})

        for P in [0, 1]:
            for Q in [0, 1]:
                if P == 0 and Q == 0:
                    continue
                fit = SARIMAX(train_r, order=BASE_ORDER,
                              seasonal_order=(P, 0, Q, SEASONAL_PERIOD)).fit(disp=False)
                rows.append({"P": P, "D": 0, "Q": Q, "m": SEASONAL_PERIOD,
                             "aic": fit.aic, "bic": fit.bic})

    table = pd.DataFrame(rows).sort_values("aic")
    print(f"SARIMA(0,0,0)x(P,0,Q,{SEASONAL_PERIOD}) seasonal grid vs. plain ARIMA(0,0,0):")
    print(table.to_string(index=False))

    best_seasonal_aic = table[table["m"] == SEASONAL_PERIOD]["aic"].min()
    baseline_aic = table[table["m"] == 0]["aic"].iloc[0]
    improvement = baseline_aic - best_seasonal_aic

    print(f"\nBaseline ARIMA(0,0,0) AIC: {baseline_aic:.3f}")
    print(f"Best seasonal candidate AIC: {best_seasonal_aic:.3f}")
    print(f"AIC improvement from adding weekly seasonal terms: {improvement:.3f}")

    out_path = PROCESSED / "sarima_seasonal_check.csv"
    table.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    if improvement < 2:  # conventional AIC rule-of-thumb threshold
        print(
            "\nDecision: weekly seasonal terms do NOT meaningfully improve AIC "
            "(improvement < 2, the conventional threshold for a meaningfully "
            "better model). SARIMA is rejected — full walk-forward evaluation "
            "was not run for it, since there is no in-sample evidence it would "
            "outperform plain ARIMA(0,0,0)/naive. This is a real, reportable "
            "finding: no exploitable weekly seasonality in EUR/USD returns, "
            "consistent with the ACF/PACF check."
        )
    else:
        print("\nDecision: seasonal terms show a meaningful AIC improvement — "
              "worth running through full walk-forward evaluation.")

    return table


if __name__ == "__main__":
    main()
