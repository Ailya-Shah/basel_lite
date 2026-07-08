"""
ECL — Forward-looking macro overlay (Vasicek single-factor).

IFRS 9 requires point-in-time, forward-looking PD. We shift the model's
through-the-cycle PD to point-in-time under each macro scenario via the Vasicek
single-factor transform, then probability-weight the scenario ECLs.

    PD_pit(Z) = Phi( ( Phi^-1(PD_ttc) + sqrt(rho) * Z ) / sqrt(1 - rho) )

Convention: Z > 0 = adverse (higher PD), Z < 0 = upside (lower PD).

PROTOTYPE NOTE: Z values here are stress assumptions, not fitted from data. The
data-driven upgrade (regress vintage default rates on a real macro series, e.g.
FRED unemployment, using issue_d) is the documented next step.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def vasicek_pit(pd_ttc: np.ndarray, Z: float, rho: float) -> np.ndarray:
    """Shift TTC PD to point-in-time under systematic factor Z."""
    pd_ttc = np.clip(np.asarray(pd_ttc, dtype=float), 1e-6, 1 - 1e-6)
    x = (norm.ppf(pd_ttc) + np.sqrt(rho) * Z) / np.sqrt(1.0 - rho)
    return norm.cdf(x)


def scenario_pds(pd_ttc: np.ndarray, scenarios: list, rho: float) -> dict:
    """
    Map each scenario name -> point-in-time PD array.
    scenarios: list of (name, Z, weight).
    """
    return {name: vasicek_pit(pd_ttc, Z, rho) for name, Z, _ in scenarios}


def weights(scenarios: list) -> dict:
    """name -> probability weight (validated to sum to 1)."""
    w = {name: wt for name, _, wt in scenarios}
    total = sum(w.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Scenario weights must sum to 1, got {total}")
    return w
