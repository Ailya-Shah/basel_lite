"""
ECL — Step 1: PD term structure.

IFRS 9 lifetime ECL needs a *marginal* PD for every future period, not one number.
Basel-Lite already produces two ingredients:

  1. the LightGBM PD  -> each loan's overall default *level* (how risky this borrower is)
  2. the survival curve -> the *timing* of defaults (when, over the life, defaults land)

This module combines them: it uses the survival curve to spread each loan's lifetime
PD across the months of its term, giving marginal PD(t) with sum_t marginal PD(t) = PD.
That's the term structure everything downstream (12-month ECL, lifetime ECL) is built on.

Assumption (stated honestly): every loan shares the *shape* of the portfolio survival
curve and differs only in level (its own PD). A fuller model would fit a per-loan Cox
survival curve so the shape varies by borrower too — that's the documented next step.
"""
from __future__ import annotations

import numpy as np


def survival_to_weights(surv_months: np.ndarray, surv_probs: np.ndarray,
                        horizon: int) -> np.ndarray:
    """
    Convert a baseline survival curve S0(t) into per-month marginal-default weights.

    weight(t) = share of lifetime defaults that occur in month t, normalised so the
    weights over 1..horizon sum to 1. Built from the drop in survival each month:
    (S0(t-1) - S0(t)).

    Parameters
    ----------
    surv_months : increasing array of month indices where the curve is defined (from t=0)
    surv_probs  : S0 at those months (starts at ~1.0, decreasing)
    horizon     : number of months to produce weights for (e.g. loan term)

    Returns
    -------
    weights : array length `horizon`, sums to 1.
    """
    surv_months = np.asarray(surv_months, dtype=float)
    surv_probs = np.asarray(surv_probs, dtype=float)

    # interpolate S0 onto a monthly grid 0..horizon (step-forward from known points)
    grid = np.arange(0, horizon + 1)
    s_grid = np.interp(grid, surv_months, surv_probs, left=1.0,
                       right=surv_probs[-1])
    s_grid = np.minimum.accumulate(s_grid)          # enforce non-increasing

    marg = -np.diff(s_grid)                          # S0(t-1) - S0(t), length horizon
    marg = np.clip(marg, 0, None)
    total = marg.sum()
    if total <= 0:
        # degenerate curve -> spread evenly rather than divide by zero
        return np.full(horizon, 1.0 / horizon)
    return marg / total


def loan_pd_schedule(pd_lifetime: float, term_months: int,
                     weights: np.ndarray) -> np.ndarray:
    """
    Marginal PD per month for one loan: pd_lifetime * weight(t), t = 1..term_months.
    sum over the schedule == pd_lifetime (by construction).
    """
    w = weights[:term_months]
    w = w / w.sum() if w.sum() > 0 else np.full(term_months, 1.0 / term_months)
    return float(pd_lifetime) * w


def pd_12m(schedule: np.ndarray) -> float:
    """12-month PD = sum of the first 12 monthly marginals (for Stage 1 / 12-month ECL)."""
    return float(np.sum(schedule[:12]))


def survival_from_schedule(schedule: np.ndarray) -> np.ndarray:
    """
    Cumulative survival implied by a marginal-PD schedule: S(t) = 1 - cumsum(marginal).
    Used later for discounting and EAD (a loan can only default if it survived to t).
    """
    return 1.0 - np.cumsum(schedule)
