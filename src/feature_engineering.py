"""
Build the feature table for XGBoost/LSTM modeling from the cleaned daily
FX series, and produce a chronological train/validation/test split.

Every feature is computed on the full, continuously-sorted date series
BEFORE splitting, using .shift(1) ahead of any rolling window. This is
deliberate and not a leakage risk: for a feature at row t, .shift(1)
means the window only ever touches r_{t-1}, r_{t-2}, ... — days strictly
before the day being predicted — regardless of which split that row later
falls into. Computing on the full series (rather than per-split) is what
lets val/test rows have valid lag/rolling values from day one of each
split, instead of an artificial NaN warm-up period at every split
boundary.

Reusable across pairs: point --input at any cleaned data_pull.py output.

Usage:
    python src/feature_engineering.py --input data/processed/eurusd_daily.csv \
        --prefix eurusd
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent

LAG_DEPTH = 5  # one full trading week (Mon-Fri) — see DECISIONS_AND_ISSUES_LOG.md
SHORT_WINDOW = 5  # 1 trading week
LONG_WINDOW = 20  # ~1 trading month

# Feature columns used by the ML/DL models (excludes date/close/log_return,
# which are kept for reference/reconstruction, not fed to the models as-is).
FEATURE_COLUMNS = (
    [f"lag_{i}" for i in range(1, LAG_DEPTH + 1)]
    + [f"roll_mean_{SHORT_WINDOW}", f"roll_std_{SHORT_WINDOW}"]
    + [f"roll_mean_{LONG_WINDOW}", f"roll_std_{LONG_WINDOW}"]
    + ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri"]
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    for i in range(1, LAG_DEPTH + 1):
        df[f"lag_{i}"] = df["log_return"].shift(i)

    # Rolling stats: shift(1) FIRST, then roll — the window for row t covers
    # r_{t-1} .. r_{t-window}, never r_t itself.
    shifted_return = df["log_return"].shift(1)
    df[f"roll_mean_{SHORT_WINDOW}"] = shifted_return.rolling(SHORT_WINDOW).mean()
    df[f"roll_std_{SHORT_WINDOW}"] = shifted_return.rolling(SHORT_WINDOW).std()
    df[f"roll_mean_{LONG_WINDOW}"] = shifted_return.rolling(LONG_WINDOW).mean()
    df[f"roll_std_{LONG_WINDOW}"] = shifted_return.rolling(LONG_WINDOW).std()

    # Day-of-week is a calendar fact known in advance of the close, not a
    # backward-looking statistic — no shift needed. One-hot encoded since
    # weekday order (Mon..Fri) has no meaningful ordinal distance for a
    # tree/NN model. Only 5 levels ever appear (no weekend rows in the data).
    dow = df["date"].dt.dayofweek  # 0=Mon ... 4=Fri
    dow_names = ["mon", "tue", "wed", "thu", "fri"]
    for i, name in enumerate(dow_names):
        df[f"dow_{name}"] = (dow == i).astype(int)

    return df


def assert_no_lookahead_leakage(df: pd.DataFrame) -> None:
    """Sanity check: every feature column's value at row t must be
    reproducible from log_return[0:t] alone (never log_return[t] or later).
    Recomputes lag_1 and roll_mean_5 independently and compares.
    """
    check = df["log_return"].shift(1)
    assert (df["lag_1"].fillna(-999).values == check.fillna(-999).values).all(), \
        "lag_1 leakage check failed"

    check_roll = df["log_return"].shift(1).rolling(SHORT_WINDOW).mean()
    a, b = df[f"roll_mean_{SHORT_WINDOW}"].fillna(-999), check_roll.fillna(-999)
    assert np.allclose(a.values, b.values), "roll_mean leakage check failed"
    print("Leakage check passed: lag_1 and roll_mean_5 independently reproduced "
          "from log_return.shift(1)-based windows only.")


def chronological_split(df: pd.DataFrame, train_end: str, val_end: str) -> dict:
    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]
    return {"train": train, "val": val, "test": test}


def scale_features(splits: dict, feature_cols: list) -> tuple:
    scaler = StandardScaler()
    scaler.fit(splits["train"][feature_cols])

    scaled = {}
    for name, part in splits.items():
        scaled_values = scaler.transform(part[feature_cols])
        scaled_df = part.copy()
        scaled_cols = [f"{c}_scaled" for c in feature_cols]
        scaled_df[scaled_cols] = scaled_values
        scaled[name] = scaled_df
    return scaled, scaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/eurusd_daily.csv")
    parser.add_argument("--prefix", default="eurusd")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--val-end", default="2023-12-31")
    args = parser.parse_args()

    df = pd.read_csv(REPO_ROOT / args.input, parse_dates=["date"])
    df = build_features(df)
    assert_no_lookahead_leakage(df)

    # Drop the warm-up rows at the very start of the series where the
    # longest rolling window (20 days) isn't yet fully populated. This only
    # affects the first ~20 rows of the TRAIN split (1999), since features
    # were computed on the full continuous series before splitting.
    n_before = len(df)
    df_model = df.dropna(subset=FEATURE_COLUMNS + ["log_return"]).reset_index(drop=True)
    n_after = len(df_model)
    print(f"Dropped {n_before - n_after} warm-up/undefined rows "
          f"(need {LONG_WINDOW} prior days for the longest rolling window).")

    splits = chronological_split(df_model, args.train_end, args.val_end)
    for name, part in splits.items():
        print(f"{name:5s}: {part['date'].min().date()} -> {part['date'].max().date()} "
              f"({len(part)} rows, {len(part) / len(df_model) * 100:.1f}%)")

    scaled_splits, scaler = scale_features(splits, FEATURE_COLUMNS)

    processed_dir = REPO_ROOT / "data" / "processed"
    for name, part in scaled_splits.items():
        out_path = processed_dir / f"{args.prefix}_{name}.csv"
        part.to_csv(out_path, index=False, date_format="%Y-%m-%d")
        print(f"Saved {out_path}")

    import joblib
    scaler_path = processed_dir / f"{args.prefix}_feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted StandardScaler (fit on train only) to {scaler_path}")

    metadata = {
        "prefix": args.prefix,
        "feature_columns": FEATURE_COLUMNS,
        "lag_depth": LAG_DEPTH,
        "short_window": SHORT_WINDOW,
        "long_window": LONG_WINDOW,
        "train_end": args.train_end,
        "val_end": args.val_end,
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "split_date_ranges": {
            name: [str(part["date"].min().date()), str(part["date"].max().date())]
            for name, part in splits.items()
        },
    }
    meta_path = processed_dir / f"{args.prefix}_split_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved split metadata to {meta_path}")


if __name__ == "__main__":
    main()
