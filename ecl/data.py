"""
ECL — data loading and scoring.

Reuses Basel-Lite's saved artifacts and the same PD scoring path as backend.py /
the validator, so the ECL engine sees identical PDs to everything else.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C

from dotenv import load_dotenv
load_dotenv(C.REPO_ROOT / ".env", override=True)  # pick up BASEL_DB_URL like the notebook does


def load_artifacts() -> dict:
    missing = [str(p) for p in C.ARTIFACTS.values() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing artifacts (run basel_lite.ipynb first):\n  "
                                + "\n  ".join(missing))
    art = {k: joblib.load(v) for k, v in C.ARTIFACTS.items()}
    meta = art["meta"]
    art["features"] = list(meta["features"])
    art["avg_lgd"] = float(meta.get("avg_lgd", C.FALLBACK_LGD))
    return art


def load_frame() -> pd.DataFrame:
    """loans_clean, from an offline file if set, else the DB (same as the validator)."""
    data_file = os.getenv(C.DATA_FILE_ENV, "").strip()
    if data_file:
        p = Path(data_file)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    else:
        db_url = os.getenv(C.DB_URL_ENV)
        if not db_url:
            raise RuntimeError(f"Set {C.DATA_FILE_ENV}=<file> or {C.DB_URL_ENV}.")
        from sqlalchemy import create_engine
        df = pd.read_sql(f"SELECT * FROM {C.TABLE}", create_engine(db_url))
    if C.SAMPLE_N and C.SAMPLE_N < len(df):
        df = df.sample(C.SAMPLE_N, random_state=42).reset_index(drop=True)
    return df


def pd_lifetime(art: dict, df: pd.DataFrame) -> np.ndarray:
    """Calibrated lifetime PD — identical scoring to backend.py."""
    woe = art["binning"].transform(df[art["features"]], metric="woe")
    return art["pd_model"].predict_proba(woe)[:, 1]


def term_months(df: pd.DataFrame) -> np.ndarray:
    """Parse ' 36 months' / ' 60 months' -> int months."""
    return (df["term"].astype(str).str.extract(r"(\d+)")[0]
            .astype(float).fillna(36).astype(int).to_numpy())


def load_survival_curve() -> tuple[np.ndarray, np.ndarray]:
    """
    Baseline survival S0(t). Prefer a saved Kaplan-Meier artifact; otherwise fall
    back to a parametric shape (only the SHAPE matters — weights are normalised).

    To save the real curve from basel_lite.ipynb (after the KM fit), add:
        import joblib
        joblib.dump({"months": km.survival_function_.index.to_numpy(),
                     "survival": km.survival_function_.iloc[:,0].to_numpy()},
                    "assets/models/survival_curve.joblib")
    """
    if Path(C.SURVIVAL_CURVE).exists():
        d = joblib.load(C.SURVIVAL_CURVE)
        return np.asarray(d["months"], float), np.asarray(d["survival"], float)

    # Parametric fallback: hazard rises then tapers (typical loan seasoning).
    months = np.arange(0, 61)
    # discrete hazard peaking ~month 15, integrated into a survival curve
    hz = 0.004 * np.exp(-0.5 * ((months - 15) / 10.0) ** 2)
    surv = np.cumprod(1.0 - hz)
    surv[0] = 1.0
    return months, surv