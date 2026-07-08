"""
ECL — EAD schedule.

Exposure at default isn't flat: an amortizing loan's outstanding balance falls
every month. This replaces Basel-Lite's "EAD = loan amount" shortcut with a real
amortization schedule, so later months carry less exposure — which matters for
lifetime ECL, where most default risk lands after the balance has already dropped.
"""
from __future__ import annotations

import numpy as np


def amortization_ead(principal: float, annual_rate_pct: float,
                     term_months: int) -> np.ndarray:
    """
    Outstanding balance at the START of each month, t = 1..term_months.

    Standard fixed-payment amortization. EAD for month t = balance still owed
    when month t begins (i.e. balance after t-1 payments).

    Returns array length `term_months`; element 0 == full principal.
    """
    principal = float(principal)
    r = annual_rate_pct / 100.0 / 12.0            # monthly rate
    n = int(term_months)

    k = np.arange(n)                               # 0..n-1  (start-of-month index)
    if r <= 0:
        # straight-line paydown if no interest
        bal_start = principal * (1.0 - k / n)
    else:
        # balance after k payments: P * ((1+r)^n - (1+r)^k) / ((1+r)^n - 1)
        g = (1.0 + r)
        bal_start = principal * (g**n - g**k) / (g**n - 1.0)
    return np.clip(bal_start, 0.0, principal)


def discount_factors(annual_rate_pct: float, term_months: int,
                     use_eir: bool = True) -> np.ndarray:
    """
    Monthly discount factors DF(t) = 1 / (1+r)^t, t = 1..term_months.
    IFRS 9 discounts at the effective interest rate; set use_eir=False for no
    discounting (DF = 1) if you want to compare.
    """
    n = int(term_months)
    if not use_eir:
        return np.ones(n)
    r = annual_rate_pct / 100.0 / 12.0
    t = np.arange(1, n + 1)
    return 1.0 / (1.0 + r) ** t
