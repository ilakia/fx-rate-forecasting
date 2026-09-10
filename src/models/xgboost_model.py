"""
XGBoost model for an FX pair's log-return series, using the engineered
feature table from src/feature_engineering.py (raw, unscaled columns —
tree splits are scale-invariant, so the *_scaled columns built for LSTM
aren't needed here). Originally built for EUR/USD (Stage 4); parameterized
in Stage 6 to reapply unchanged to GBP/USD and USD/JPY via --prefix —
each pair gets its own independent hyperparameter grid search rather
than reusing EUR/USD's selected configuration.

Unlike ARIMA (src/models/arima_sarima.py), this model is NOT evaluated
with a walk-forward-with-periodic-refit loop. That's a deliberate,
justified difference, not an inconsistency or a shortcut — see the
module-level note below and DECISIONS_AND_ISSUES_LOG.md for the full
reasoning:

ARIMA's own internal state (an estimated mean, or AR/MA coefficients)
is a summary of everything the model has seen so far, and that summary
can go stale as new data arrives — which is exactly why walk-forward
evaluation re-exposes it to fresh true values one step at a time and
periodically refits. XGBoost's features, by contrast, are already
genuine backward-looking historical values (real past returns, real
past rolling stats) computed once for every row in the table — a
validation or test row's features already encode "true data known up to
that point" regardless of when the model itself was fit. A single fit
on train, then a plain one-shot predict on val/test, does not leak
future information and does not need walk-forward's rolling-origin
machinery to stay honest.

Hyperparameter tuning: a small grid search over regularization-relevant
parameters (max_depth, min_child_weight, subsample, colsample_bytree,
learning_rate), each combination trained with early stopping on the
VALIDATION set's RMSE (never test). Given how weak the ARIMA signal was,
every configuration is also checked for a train/val performance gap
before anything is treated as a genuine result.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.evaluate import evaluate, results_to_table
from feature_engineering import FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"

PARAM_GRID = {
    "max_depth": [2, 3],
    "min_child_weight": [10, 20],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "learning_rate": [0.01, 0.05],
}
FIXED_PARAMS = dict(
    n_estimators=1000,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    objective="reg:squarederror",
)
EARLY_STOPPING_ROUNDS = 50


def load_splits(prefix: str):
    train = pd.read_csv(PROCESSED / f"{prefix}_train.csv", parse_dates=["date"])
    val = pd.read_csv(PROCESSED / f"{prefix}_val.csv", parse_dates=["date"])
    test = pd.read_csv(PROCESSED / f"{prefix}_test.csv", parse_dates=["date"])
    return train, val, test


def grid_search(X_train, y_train, X_val, y_val):
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    rows = []
    best = None

    for combo in combos:
        params = dict(zip(keys, combo))
        model = xgb.XGBRegressor(
            **FIXED_PARAMS,
            **params,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            eval_metric="rmse",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        best_iter = model.best_iteration
        train_rmse = model.evals_result()["validation_0"]["rmse"][best_iter]
        val_rmse = model.evals_result()["validation_1"]["rmse"][best_iter]
        row = {**params, "best_iteration": best_iter,
               "train_rmse": train_rmse, "val_rmse": val_rmse,
               "train_val_gap": val_rmse - train_rmse}
        rows.append(row)
        if best is None or val_rmse < best["val_rmse"]:
            best = {**row, "model": model}

    results = pd.DataFrame(rows).sort_values("val_rmse")
    return results, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="eurusd", help="e.g. eurusd, gbpusd, usdjpy")
    args = parser.parse_args()
    prefix = args.prefix

    train, val, test = load_splits(prefix)

    X_train, y_train = train[FEATURE_COLUMNS], train["log_return"]
    X_val, y_val = val[FEATURE_COLUMNS], val["log_return"]
    X_test, y_test = test[FEATURE_COLUMNS], test["log_return"]

    print(f"Grid search over {len(list(itertools.product(*PARAM_GRID.values())))} "
          f"hyperparameter combinations, early stopping on validation RMSE "
          f"(patience={EARLY_STOPPING_ROUNDS})...")
    grid_results, best = grid_search(X_train, y_train, X_val, y_val)
    print("\nTop 5 configurations by validation RMSE:")
    print(grid_results.head(5).to_string(index=False))

    grid_path = PROCESSED / f"{prefix}_xgboost_grid_search.csv"
    grid_results.to_csv(grid_path, index=False)
    print(f"\nSaved full grid search results to {grid_path}")

    best_model = best["model"]
    best_params = {k: best[k] for k in PARAM_GRID.keys()}
    print(f"\nSelected hyperparameters: {best_params}")
    print(f"Best iteration (early-stopped): {best['best_iteration']}")
    print(f"Train RMSE at best iteration: {best['train_rmse']:.6f}")
    print(f"Val RMSE at best iteration:   {best['val_rmse']:.6f}")

    # NOTE: raw (val_rmse - train_rmse) is NOT a clean overfitting signal here
    # — train (1999-2019) and val (2020-2023) have genuinely different
    # unconditional return volatility (train includes the 2008 crisis), so
    # their RMSEs differ even for a trivial/no-signal model. The correct
    # overfitting check is comparing each period's model RMSE against THAT
    # SAME period's own naive-baseline RMSE, done below.
    naive_train_rmse = np.sqrt(np.mean(y_train.values ** 2))
    naive_val_rmse = np.sqrt(np.mean(y_val.values ** 2))
    print(f"\nOverfitting check (model RMSE vs. naive RMSE, same period each time):")
    print(f"  Train: model {best['train_rmse']:.6f} vs. naive {naive_train_rmse:.6f} "
          f"(ratio {best['train_rmse'] / naive_train_rmse:.4f})")
    print(f"  Val:   model {best['val_rmse']:.6f} vs. naive {naive_val_rmse:.6f} "
          f"(ratio {best['val_rmse'] / naive_val_rmse:.4f})")
    if best["train_rmse"] / naive_train_rmse < 0.98 and best["val_rmse"] / naive_val_rmse >= 0.98:
        print("  -> Model beats naive on train but not val: classic overfitting signature.")
    elif best["val_rmse"] < naive_val_rmse * 0.98:
        print("  -> XGBoost val RMSE is notably better than naive — flagging for "
              "extra scrutiny before accepting this as a real result.")
    else:
        print("  -> Model matches naive on BOTH train and val (ratios ~1.0) — "
              "early stopping converged to essentially the trivial/no-signal "
              "prediction on every period, not a model that overfit train and "
              "failed to generalize. Consistent with the ARIMA finding.")

    # --- Feature importance (gain-based) ---
    # IMPORTANT: booster.get_score() aggregates gain across every tree ever
    # built during training (up to early_stopping_rounds past the best one),
    # not just the trees actually used for prediction. With early stopping,
    # predict() only uses trees [0, best_iteration] — so naively calling
    # get_score() can attribute "importance" to features that only appear in
    # discarded, never-deployed later trees. Restrict explicitly to the
    # deployed tree range via trees_to_dataframe() instead. See
    # DECISIONS_AND_ISSUES_LOG.md for how this was caught.
    booster = best_model.get_booster()
    trees_df = booster.trees_to_dataframe()
    deployed = trees_df[trees_df["Tree"] <= best["best_iteration"]]
    deployed_splits = deployed[deployed["Feature"] != "Leaf"]
    gain_by_feature = deployed_splits.groupby("Feature")["Gain"].sum().to_dict()
    importance_full = {f: gain_by_feature.get(f, 0.0) for f in FEATURE_COLUMNS}
    importance_df = (
        pd.DataFrame(list(importance_full.items()), columns=["feature", "gain"])
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    n_trees_deployed = best["best_iteration"] + 1
    n_trees_attempted = booster.num_boosted_rounds()
    print(f"\nDeployed model uses {n_trees_deployed} tree(s) out of "
          f"{n_trees_attempted} attempted before early stopping — feature "
          f"importance below is restricted to those {n_trees_deployed} "
          f"deployed tree(s) only.")
    print("\nFeature importance (gain-based):")
    print(importance_df.to_string(index=False))

    dow_cols = [c for c in FEATURE_COLUMNS if c.startswith("dow_")]
    dow_total_gain = importance_df[importance_df["feature"].isin(dow_cols)]["gain"].sum()
    vol_cols = ["roll_std_5", "roll_std_20"]
    vol_total_gain = importance_df[importance_df["feature"].isin(vol_cols)]["gain"].sum()
    total_gain = importance_df["gain"].sum()
    print(f"\nDay-of-week features: {dow_total_gain:.2f} / {total_gain:.2f} total gain "
          f"({dow_total_gain / total_gain * 100 if total_gain else 0:.1f}%)")
    print(f"Rolling-volatility features (roll_std_5, roll_std_20): "
          f"{vol_total_gain:.2f} / {total_gain:.2f} total gain "
          f"({vol_total_gain / total_gain * 100 if total_gain else 0:.1f}%)")

    imp_path = PROCESSED / f"{prefix}_xgboost_feature_importance.csv"
    importance_df.to_csv(imp_path, index=False)
    print(f"Saved feature importance to {imp_path}")

    # --- Evaluate through the shared module ---
    def build_eval(name, period, df, X):
        pred_return = best_model.predict(X)
        true_return = df["log_return"].values
        true_price = df["close"].values
        prev_price = true_price / np.exp(true_return)
        return evaluate(name, period, true_return, pred_return, prev_price, true_price)

    results = [
        build_eval("XGBoost", "val", val, X_val),
        build_eval("XGBoost", "test", test, X_test),
    ]
    table = results_to_table(results)
    print("\nXGBoost evaluation results:")
    print(table.to_string(index=False))

    out_path = PROCESSED / f"{prefix}_xgboost_results.csv"
    table.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    best_model.save_model(str(PROCESSED / f"{prefix}_xgboost_model.json"))
    params_out = {"best_hyperparameters": best_params, "fixed_params": FIXED_PARAMS,
                  "best_iteration": int(best["best_iteration"]),
                  "train_rmse": float(best["train_rmse"]),
                  "val_rmse": float(best["val_rmse"])}
    (PROCESSED / f"{prefix}_xgboost_selected_params.json").write_text(json.dumps(params_out, indent=2))

    return table


if __name__ == "__main__":
    main()
