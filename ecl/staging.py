"""
ECL — Staging (IFRS 9 Stage 1 / 2 / 3).

Stage decides the ECL horizon:
  Stage 1  (performing)         -> 12-month ECL
  Stage 2  (significant risk)   -> lifetime ECL
  Stage 3  (credit-impaired)    -> lifetime ECL

HONEST LIMITATION: real staging tests for a *significant increase in credit risk
since origination* (SICR) — PD now vs PD at origination. Static LendingClub data
has no such history, so this is a PROXY based on absolute risk at scoring time
(PD level + delinquency), not a true origination comparison. The report states this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_stages(pd_hat: np.ndarray, df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Return an int array of stages (1/2/3), one per loan.

    cfg keys: stage3_pd, stage2_pd, use_delinquency, delinq_col.
    """
    pd_hat = np.asarray(pd_hat, dtype=float)
    stage = np.ones(len(pd_hat), dtype=int)

    stage[pd_hat >= cfg["stage2_pd"]] = 2
    stage[pd_hat >= cfg["stage3_pd"]] = 3

    if cfg.get("use_delinquency") and cfg["delinq_col"] in df.columns:
        delinq = pd.to_numeric(df[cfg["delinq_col"]], errors="coerce").fillna(0).to_numpy()
        bump = (delinq >= 1) & (stage < 2)     # delinquency forces at least Stage 2
        stage[bump] = 2
    return stage


def stage_summary(stage: np.ndarray) -> dict:
    """Count and share of loans in each stage."""
    out = {}
    n = len(stage)
    for s in (1, 2, 3):
        c = int((stage == s).sum())
        out[f"stage_{s}"] = {"count": c, "share": (c / n if n else 0.0)}
    return out
