"""
Shared evaluation module — every model in this project (naive baseline,
ARIMA/SARIMA, and later XGBoost/LSTM) calls this to get scored, so
comparisons across models are apples-to-apples rather than each model
inventing its own metric logic.

Metrics computed:
- RMSE / MAE on the log-return scale (the actual modeling target).
- RMSE / MAE reconstructed to price level, using the TRUE previous price
  at each step (P_hat_t = P_{t-1} * exp(r_hat_t)) — not a chained/
  compounded forecast where each day's prediction builds on the model's
  own previous prediction. This is a deliberate choice: chaining lets
  small early errors compound into large, misleading late-period errors
  that reflect accumulated drift more than actual next-day forecast
  skill. Reconstructing from the true previous price isolates each
  day's forecast quality independently, matching the one-step-ahead
  walk-forward evaluation philosophy used throughout this project. See
  DECISIONS_AND_ISSUES_LOG.md for the full reasoning.
- Directional accuracy: % of days where sign(predicted return) matches
  sign(actual return). Zero-return ties are EXCLUDED from the
  denominator (see `direction_ties` policy below) rather than counted
  as automatic hits or misses, since a model outputting exactly 0.0 is
  making no directional claim at all — scoring it as right or wrong
  either way would misrepresent what the model actually did. This
  policy is applied identically to every model, including the naive
  baseline, whose predictions are 0.0 every single day (see the naive
  baseline's own writeup for why this makes its directional accuracy
  degenerate/undefined by construction, not a modeling result).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EvalResult:
    model_name: str
    period: str  # "val" or "test"
    n_obs: int
    rmse_return: float
    mae_return: float
    rmse_price: float
    mae_price: float
    directional_accuracy: Optional[float]  # None if undefined (no non-tie predictions)
    n_direction_ties_excluded: int
    n_direction_evaluated: int
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "model": self.model_name,
            "period": self.period,
            "n_obs": self.n_obs,
            "rmse_return": self.rmse_return,
            "mae_return": self.mae_return,
            "rmse_price": self.rmse_price,
            "mae_price": self.mae_price,
            "directional_accuracy": self.directional_accuracy,
            "n_direction_ties_excluded": self.n_direction_ties_excluded,
        }


def _reconstruct_price(prev_price: np.ndarray, pred_return: np.ndarray) -> np.ndarray:
    """P_hat_t = P_{t-1} * exp(r_hat_t), using the TRUE previous price at
    every step (never a previously predicted price) — see module docstring.
    """
    return prev_price * np.exp(pred_return)


def _directional_accuracy(true_return: np.ndarray, pred_return: np.ndarray) -> tuple:
    """Returns (accuracy_or_None, n_ties_excluded, n_evaluated).

    Tie policy: a day is excluded from the denominator if EITHER the true
    return or the predicted return is exactly 0.0 — in both cases there is
    no defined "direction" to compare against. This is a stricter reading
    than "only exclude if predicted == 0", chosen because an actual
    return of exactly 0.0 (which does occur in this data — see
    DECISIONS_AND_ISSUES_LOG.md #4 on the 55 flat-close days) is equally
    undefined for direction-matching purposes.
    """
    true_sign = np.sign(true_return)
    pred_sign = np.sign(pred_return)
    is_tie = (true_sign == 0) | (pred_sign == 0)
    n_ties = int(is_tie.sum())
    n_eval = int((~is_tie).sum())
    if n_eval == 0:
        return None, n_ties, n_eval
    matches = (true_sign[~is_tie] == pred_sign[~is_tie]).sum()
    return float(matches / n_eval), n_ties, n_eval


def evaluate(
    model_name: str,
    period: str,
    true_return: np.ndarray,
    pred_return: np.ndarray,
    prev_price: np.ndarray,
    true_price: np.ndarray,
    extra: Optional[dict] = None,
) -> EvalResult:
    """
    Parameters
    ----------
    true_return, pred_return : arrays of log returns for the period, aligned 1:1.
    prev_price : the TRUE price at t-1 for each row (used to reconstruct
        both true_price and predicted price on the same basis, so the
        price-level RMSE/MAE comparison isn't confounded by using a
        different reconstruction convention for actual vs. predicted).
    true_price : the TRUE price at t for each row (i.e. prev_price * exp(true_return),
        passed in explicitly rather than recomputed, so it matches whatever
        the source data's actual recorded close was).
    """
    true_return = np.asarray(true_return, dtype=float)
    pred_return = np.asarray(pred_return, dtype=float)
    prev_price = np.asarray(prev_price, dtype=float)
    true_price = np.asarray(true_price, dtype=float)

    n = len(true_return)
    assert len(pred_return) == n and len(prev_price) == n and len(true_price) == n, \
        "evaluate(): all input arrays must be the same length and aligned"

    resid_return = true_return - pred_return
    rmse_return = float(np.sqrt(np.mean(resid_return ** 2)))
    mae_return = float(np.mean(np.abs(resid_return)))

    pred_price = _reconstruct_price(prev_price, pred_return)
    resid_price = true_price - pred_price
    rmse_price = float(np.sqrt(np.mean(resid_price ** 2)))
    mae_price = float(np.mean(np.abs(resid_price)))

    dir_acc, n_ties, n_eval = _directional_accuracy(true_return, pred_return)

    return EvalResult(
        model_name=model_name,
        period=period,
        n_obs=n,
        rmse_return=rmse_return,
        mae_return=mae_return,
        rmse_price=rmse_price,
        mae_price=mae_price,
        directional_accuracy=dir_acc,
        n_direction_ties_excluded=n_ties,
        n_direction_evaluated=n_eval,
        extra=extra or {},
    )


def results_to_table(results: list) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame([r.as_row() for r in results])
