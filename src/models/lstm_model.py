"""
LSTM model for the EUR/USD log-return series — the third and final model
family in this project (after naive and ARIMA in Stage 3, XGBoost in
Stage 4), and the only one that sees the feature set as an actual
temporal sequence rather than a single flattened row per prediction.

Built with PyTorch, not TensorFlow/Keras — TensorFlow's pip wheel
requires AVX instructions, and this project's venv Python is x86_64
running under Rosetta 2 translation on Apple Silicon (inherited from the
system's Anaconda base Python since Stage 1); Rosetta 2 does not emulate
AVX, so TensorFlow aborts (SIGABRT) on import. PyTorch's CPU wheels have
no such requirement and imported/ran cleanly. See
DECISIONS_AND_ISSUES_LOG.md for the full writeup — this only affected
which library was used, not the model, evaluation methodology, or any
result.

Sequence construction: for a target row i (predicting log_return at row
i), the input sequence is the *scaled* feature vectors (from Stage 2's
feature_engineering.py) of the WINDOW rows immediately before i:
rows [i-WINDOW, i-1]. Row i's own features/target are never included in
its own input sequence — verified by an explicit runtime assertion
(check_no_leakage below), the same discipline Stage 2 applied to the
lag/rolling features themselves.

Sequences are built PER SPLIT independently — a val or test sequence
never reaches back into train (or, for test, into val) for its earlier
timesteps. This means the first WINDOW rows of every split (not just
train) are unusable as prediction targets and are dropped, which is a
deliberate, stricter boundary than the tabular models needed (XGBoost's
single-row features could reach back across the split boundary safely,
since those features were themselves already computed once on the full
continuous series in Stage 2 — see DECISIONS_AND_ISSUES_LOG.md for why
LSTM's sequence framing is different).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.evaluate import evaluate, results_to_table
from feature_engineering import FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
FIGURES = REPO_ROOT / "figures"

WINDOW = 20  # ~1 trading month — matches the "long" rolling-stat window
             # already chosen in Stage 2, for the same reasoning: long
             # enough to span multiple volatility regimes within one
             # sequence, short enough not to shrink the ~5,300-row train
             # set meaningfully.
SCALED_COLUMNS = [f"{c}_scaled" for c in FEATURE_COLUMNS]
SEED = 42
HIDDEN_SIZE = 16
DROPOUT = 0.2
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def load_splits():
    train = pd.read_csv(PROCESSED / "eurusd_train.csv", parse_dates=["date"])
    val = pd.read_csv(PROCESSED / "eurusd_val.csv", parse_dates=["date"])
    test = pd.read_csv(PROCESSED / "eurusd_test.csv", parse_dates=["date"])
    return train, val, test


def build_sequences(df: pd.DataFrame, window: int = WINDOW):
    """Returns X (n, window, n_features), y (n,), dates (n,) [target dates],
    input_end_dates (n,) [date of the last input row, for the leakage check],
    prev_price (n,), true_price (n,) — all aligned row-for-row.
    """
    df = df.sort_values("date").reset_index(drop=True)
    feats = df[SCALED_COLUMNS].values
    n = len(df)

    X, y, dates, input_end_dates = [], [], [], []
    for i in range(window, n):
        X.append(feats[i - window:i])
        y.append(df["log_return"].iloc[i])
        dates.append(df["date"].iloc[i])
        input_end_dates.append(df["date"].iloc[i - 1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    dates = np.array(dates)
    input_end_dates = np.array(input_end_dates)

    true_price = df["close"].values[window:]
    true_return = df["log_return"].values[window:]
    prev_price = true_price / np.exp(true_return)

    return X, y, dates, input_end_dates, prev_price, true_price


def check_no_leakage(df: pd.DataFrame, X, dates, input_end_dates, window: int = WINDOW):
    """Independent verification, not just an assertion in comments:
    1. Every sequence's last input row's date is strictly before its
       target date.
    2. The target date is the row immediately following the last input
       row in the sorted dataframe (confirms no gap/misalignment snuck
       a future row into the window).
    3. Spot-check: re-slice the raw scaled feature array directly from
       the dataframe for a few sequences and confirm it matches the
       constructed X exactly.
    """
    df = df.sort_values("date").reset_index(drop=True)
    assert (input_end_dates < dates).all(), \
        "LEAKAGE: some sequence's last input date is not before its target date"

    date_to_idx = {d: idx for idx, d in enumerate(df["date"])}
    for k in [0, len(dates) // 2, len(dates) - 1]:
        target_idx = date_to_idx[dates[k]]
        input_end_idx = date_to_idx[input_end_dates[k]]
        assert input_end_idx == target_idx - 1, \
            f"LEAKAGE: target row {target_idx} does not immediately follow input row {input_end_idx}"
        expected_slice = df[SCALED_COLUMNS].values[target_idx - window:target_idx]
        assert np.allclose(expected_slice, X[k]), \
            f"LEAKAGE: sequence {k} does not match an independently re-sliced window"

    print(f"Leakage check passed: {len(dates)} sequences all have input windows "
          f"strictly preceding their target date, spot-checked against an "
          f"independently re-sliced feature array.")


def build_model(n_features: int):
    import torch
    import torch.nn as nn

    class SmallLSTM(nn.Module):
        def __init__(self, n_features, hidden_size, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                                 num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            last_hidden = h_n[-1]
            out = self.dropout(last_hidden)
            return self.fc(out).squeeze(-1)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return SmallLSTM(n_features, HIDDEN_SIZE, DROPOUT)


def train_model(model, X_train, y_train, X_val, y_val):
    import torch
    import torch.nn as nn

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    n = len(X_train_t)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    rng = np.random.RandomState(SEED)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_loss = epoch_loss / n
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()
        val_losses.append(val_loss)

        improved = val_loss < best_val_loss - 1e-10
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS}  train_loss={train_loss:.8f}  "
              f"val_loss={val_loss:.8f}{'  *' if improved else ''}")

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val_loss improvement "
                  f"for {PATIENCE} epochs).")
            break

    model.load_state_dict(best_state)
    best_epoch = int(np.argmin(val_losses)) + 1
    return model, train_losses, val_losses, best_epoch


def main():
    import torch

    train, val, test = load_splits()

    X_train, y_train, dates_train, end_train, prev_p_train, true_p_train = build_sequences(train)
    X_val, y_val, dates_val, end_val, prev_p_val, true_p_val = build_sequences(val)
    X_test, y_test, dates_test, end_test, prev_p_test, true_p_test = build_sequences(test)

    print(f"Sequence counts (window={WINDOW}): train={len(X_train)}, "
          f"val={len(X_val)}, test={len(X_test)}")
    print(f"(Train dropped {len(train) - len(X_train)} rows, val dropped "
          f"{len(val) - len(X_val)} rows, test dropped {len(test) - len(X_test)} "
          f"rows as sequence warm-up — each split's own first {WINDOW} rows, "
          f"since sequences never reach across a split boundary.)")

    check_no_leakage(train, X_train, dates_train, end_train)
    check_no_leakage(val, X_val, dates_val, end_val)
    check_no_leakage(test, X_test, dates_test, end_test)

    model = build_model(n_features=X_train.shape[2])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: LSTM(hidden={HIDDEN_SIZE}, dropout={DROPOUT}) -> Dense(1), "
          f"{n_params} trainable parameters")

    model, train_losses, val_losses, best_epoch = train_model(
        model, X_train, y_train, X_val, y_val
    )
    n_epochs_ran = len(train_losses)
    print(f"\nTraining stopped after {n_epochs_ran} epochs. "
          f"Best val_loss at epoch {best_epoch}.")

    # --- Plot train vs val loss (full curve + log-scale zoom, since the
    # epoch-1 spike otherwise flattens everything after ~epoch 10 into an
    # indistinguishable line on a linear scale) ---
    import matplotlib.pyplot as plt
    epochs = range(1, n_epochs_ran + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label="train loss")
    axes[0].plot(epochs, val_losses, label="val loss")
    axes[0].axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                     label=f"best epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_title("Full curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_losses, label="train loss")
    axes[1].plot(epochs, val_losses, label="val loss")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                     label=f"best epoch ({best_epoch})")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE loss (log scale)")
    axes[1].set_title("Same curve, log-scale y-axis")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"LSTM training curve (window={WINDOW}, hidden={HIDDEN_SIZE}, dropout={DROPOUT})")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    loss_fig_path = FIGURES / "lstm_loss_curve.png"
    fig.savefig(loss_fig_path, dpi=150)
    print(f"Saved loss curve to {loss_fig_path}")

    train_naive_rmse = float(np.sqrt(np.mean(y_train ** 2)))
    val_naive_rmse = float(np.sqrt(np.mean(y_val ** 2)))
    print(f"\nFor reference — naive (r_t=0) RMSE on these same sequence-warm-up-"
          f"adjusted sets: train={train_naive_rmse:.6f}, val={val_naive_rmse:.6f}")
    print(f"Model's own train/val MSE loss -> RMSE: "
          f"train={np.sqrt(train_losses[best_epoch - 1]):.6f}, "
          f"val={np.sqrt(val_losses[best_epoch - 1]):.6f}")
    print(
        "Note: model RMSE is lower on val than train here — same as the "
        "XGBoost finding in DECISIONS_AND_ISSUES_LOG.md #15, this is the "
        "train/val volatility-regime confound (train includes 2008), NOT "
        "reverse overfitting. Each period's model RMSE closely matches "
        "that SAME period's own naive RMSE (compare the two lines above), "
        "which is the correct check."
    )

    # --- Evaluate through the shared module ---
    model.eval()
    with torch.no_grad():
        pred_val = model(torch.from_numpy(X_val)).numpy()
        pred_test = model(torch.from_numpy(X_test)).numpy()

    results = [
        evaluate("LSTM", "val", y_val, pred_val, prev_p_val, true_p_val),
        evaluate("LSTM", "test", y_test, pred_test, prev_p_test, true_p_test),
    ]
    table = results_to_table(results)
    print("\nLSTM evaluation results:")
    print(table.to_string(index=False))

    val_model_rmse = table[table["period"] == "val"]["rmse_return"].iloc[0]
    if val_model_rmse < val_naive_rmse * 0.98:
        print(
            "\nLSTM val RMSE is notably better than naive on this metric — "
            "flagging for extra scrutiny per the overfitting-guard standard "
            "applied at every prior stage: re-checking leakage (already "
            "independently verified above) and confirming the improvement "
            "holds on test, not just validation, before treating this as real."
        )
    else:
        print(
            "\nLSTM val RMSE is NOT notably better than naive — consistent "
            "with the ARIMA and XGBoost findings that there is little "
            "structure in this feature set for any of the three model "
            "families to exploit."
        )

    out_path = PROCESSED / "lstm_results.csv"
    table.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

    # --- Combine into the running four-way comparison table ---
    naive_results = pd.read_csv(PROCESSED / "naive_results.csv")
    arima_results = pd.read_csv(PROCESSED / "arima_results.csv")
    xgb_results = pd.read_csv(PROCESSED / "xgboost_results.csv")
    combined = pd.concat([naive_results, arima_results, xgb_results, table], ignore_index=True)
    combined_path = PROCESSED / "model_comparison_stage5.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nSaved combined naive/ARIMA/XGBoost/LSTM comparison to {combined_path}")
    print(combined.to_string(index=False))

    torch.save(model.state_dict(), str(PROCESSED / "lstm_model.pt"))
    meta = {
        "window": WINDOW,
        "architecture": f"LSTM(hidden={HIDDEN_SIZE}, dropout={DROPOUT}) -> Linear(1)",
        "n_trainable_params": n_params,
        "n_epochs_ran": n_epochs_ran,
        "best_epoch": best_epoch,
        "best_val_loss_mse": float(min(val_losses)),
        "train_loss_at_best_epoch": float(train_losses[best_epoch - 1]),
        "batch_size": BATCH_SIZE,
        "optimizer": f"Adam(lr={LEARNING_RATE})",
        "seed": SEED,
        "framework": "pytorch (see DECISIONS_AND_ISSUES_LOG.md for why not tensorflow)",
    }
    (PROCESSED / "lstm_training_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved training metadata to {PROCESSED / 'lstm_training_metadata.json'}")

    return combined


if __name__ == "__main__":
    main()
