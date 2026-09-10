"""
Assemble one unified table comparing all three currency pairs' four-model
results (naive, ARIMA/SARIMA, XGBoost, LSTM) side by side. Reads each
pair's {prefix}_model_comparison.csv (produced by lstm_model.py, the last
model run per pair) — does not recompute anything, purely a reporting step.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"

PAIRS = [("eurusd", "EUR/USD"), ("gbpusd", "GBP/USD"), ("usdjpy", "USD/JPY")]


def main():
    frames = []
    for prefix, label in PAIRS:
        df = pd.read_csv(PROCESSED / f"{prefix}_model_comparison.csv")
        df.insert(0, "pair", label)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED / "cross_pair_comparison.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved cross-pair comparison to {out_path}\n")

    pd.set_option("display.width", 160)
    print(combined.to_string(index=False))

    # --- Did any model beat naive by a meaningful margin, on any pair? ---
    print("\n--- RMSE ratio to naive (same pair, same period) ---")
    for prefix, label in PAIRS:
        df = pd.read_csv(PROCESSED / f"{prefix}_model_comparison.csv")
        naive = df[df["model"] == "naive"].set_index("period")["rmse_return"]
        for _, row in df[df["model"] != "naive"].iterrows():
            ratio = row["rmse_return"] / naive[row["period"]]
            flag = " <-- notably better" if ratio < 0.98 else (" <-- notably worse" if ratio > 1.02 else "")
            print(f"{label:8s} {row['model']:15s} {row['period']:4s} "
                  f"ratio={ratio:.4f}{flag}")

    return combined


if __name__ == "__main__":
    main()
