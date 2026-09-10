"""
Naive "no change" baseline: predicted r_t = 0 for every day. Not fit on
any data — this is the random-walk hypothesis in its plainest form, and
the bar every other model in this project must be honestly compared
against.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.evaluate import evaluate, results_to_table

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"


def run_naive(df: pd.DataFrame, period: str):
    true_return = df["log_return"].values
    true_price = df["close"].values
    prev_price = true_price / np.exp(true_return)
    pred_return = np.zeros_like(true_return)
    return evaluate("naive", period, true_return, pred_return, prev_price, true_price)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="eurusd", help="e.g. eurusd, gbpusd, usdjpy")
    args = parser.parse_args()

    val = pd.read_csv(PROCESSED / f"{args.prefix}_val.csv", parse_dates=["date"])
    test = pd.read_csv(PROCESSED / f"{args.prefix}_test.csv", parse_dates=["date"])

    results = [run_naive(val, "val"), run_naive(test, "test")]
    table = results_to_table(results)
    print(f"Naive baseline (predict r_t = 0) results — {args.prefix}:")
    print(table.to_string(index=False))
    print(
        "\nNote on directional_accuracy: the naive baseline predicts exactly "
        "0.0 every day, which the shared tie policy in evaluate.py treats as "
        "'no direction' and excludes from the denominator entirely — so "
        "directional_accuracy is undefined (None/NaN) by construction here, "
        "not a real 0% or 100% result. RMSE/MAE are the meaningful metrics "
        "for this baseline."
    )

    out_path = PROCESSED / f"{args.prefix}_naive_results.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    return table


if __name__ == "__main__":
    main()
