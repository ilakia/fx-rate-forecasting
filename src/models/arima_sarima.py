"""
ARIMA/SARIMA model for the EUR/USD log-return series.

Order selection: pmdarima's auto_arima was attempted first but fails to
import in this environment (numpy 2.0.2 / pmdarima ABI incompatibility —
see DECISIONS_AND_ISSUES_LOG.md). Fell back to the documented manual
alternative: ACF/PACF inspection (notebooks/03_acf_pacf_check.py) followed
by a small grid search over (p, d, q) scored by AIC/BIC, fit on the
TRAINING split only. Order selection never touches validation or test.

Evaluation: one-step-ahead walk-forward forecasting through validation
then test, using true historical values up to each point (never a
multi-day-ahead compounded forecast). The model is refit periodically
(every REFIT_EVERY new observations) rather than after every single new
day, and simply extends its state (no re-estimation) in between — a
deliberate compute/accuracy tradeoff documented in the decisions log.
"""

import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.evaluate import evaluate, results_to_table

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
REFIT_EVERY = 20  # ~1 trading month; see decisions log for the tradeoff


def load_splits():
    train = pd.read_csv(PROCESSED / "eurusd_train.csv", parse_dates=["date"])
    val = pd.read_csv(PROCESSED / "eurusd_val.csv", parse_dates=["date"])
    test = pd.read_csv(PROCESSED / "eurusd_test.csv", parse_dates=["date"])
    return train, val, test


def grid_search_order(train_returns: pd.Series, p_range, d_range, q_range) -> pd.DataFrame:
    rows = []
    for p, d, q in itertools.product(p_range, d_range, q_range):
        if p == 0 and q == 0 and d == 0:
            # still worth including — this IS the "no structure" candidate
            pass
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(train_returns, order=(p, d, q))
                fit = model.fit()
            rows.append({"p": p, "d": d, "q": q, "aic": fit.aic, "bic": fit.bic})
        except Exception as e:
            rows.append({"p": p, "d": d, "q": q, "aic": np.nan, "bic": np.nan, "error": str(e)})
    return pd.DataFrame(rows).sort_values("aic")


def walk_forward(train_returns: pd.Series, wf_returns: pd.Series, order, refit_every=REFIT_EVERY):
    """One-step-ahead walk-forward forecast across wf_returns (val+test
    concatenated), starting from a model fit on train_returns. Refits
    every `refit_every` steps; otherwise extends state without refitting.
    Returns an array of predictions aligned with wf_returns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(train_returns, order=order).fit()

    preds = np.empty(len(wf_returns))
    values = wf_returns.values
    for i in range(len(values)):
        preds[i] = fit.forecast(steps=1).iloc[0]
        refit_now = ((i + 1) % refit_every == 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit.append([values[i]], refit=refit_now)
    return preds


def main():
    train, val, test = load_splits()
    train_r = train.set_index("date")["log_return"]
    val_r = val.set_index("date")["log_return"]
    test_r = test.set_index("date")["log_return"]

    print("Grid search: ARIMA(p,d,q) on train, scored by AIC/BIC")
    print("(d=1 included specifically to double-check that differencing an")
    print(" already-stationary series doesn't spuriously look better.)")
    grid = grid_search_order(train_r, p_range=range(0, 3), d_range=[0, 1], q_range=range(0, 3))
    print(grid.head(10).to_string(index=False))

    best = grid.iloc[0]
    order = (int(best["p"]), int(best["d"]), int(best["q"]))
    print(f"\nSelected order by AIC: {order}")

    grid_path = REPO_ROOT / "data" / "processed" / "arima_order_grid_search.csv"
    grid.to_csv(grid_path, index=False)
    print(f"Saved full grid search results to {grid_path}")

    # --- Walk-forward over val + test, continuous ---
    wf_returns = pd.concat([val_r, test_r])
    print(f"\nRunning one-step-ahead walk-forward over {len(wf_returns)} days "
          f"(val + test), refitting every {REFIT_EVERY} days...")
    t0 = time.time()
    preds = walk_forward(train_r, wf_returns, order)
    print(f"Walk-forward complete in {time.time() - t0:.1f}s")

    n_val = len(val_r)
    val_pred, test_pred = preds[:n_val], preds[n_val:]

    def build_eval(name, period, df, pred):
        true_price = df["close"].values
        true_return = df["log_return"].values
        prev_price = true_price / np.exp(true_return)
        return evaluate(name, period, true_return, pred, prev_price, true_price)

    results = [
        build_eval(f"ARIMA{order}", "val", val, val_pred),
        build_eval(f"ARIMA{order}", "test", test, test_pred),
    ]

    table = results_to_table(results)
    print("\nARIMA walk-forward results:")
    print(table.to_string(index=False))

    out_path = REPO_ROOT / "data" / "processed" / "arima_results.csv"
    table.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

    # Save predictions for later cross-model comparison
    preds_df = pd.DataFrame({
        "date": wf_returns.index,
        "true_return": wf_returns.values,
        "pred_return": preds,
        "split": ["val"] * n_val + ["test"] * (len(preds) - n_val),
    })
    preds_out = REPO_ROOT / "data" / "processed" / "arima_predictions.csv"
    preds_df.to_csv(preds_out, index=False, date_format="%Y-%m-%d")
    print(f"Saved predictions to {preds_out}")

    return order, table


if __name__ == "__main__":
    main()
