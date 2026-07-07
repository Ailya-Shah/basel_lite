"""
Load the trained artifacts and the holdout data, and run Basel-Lite's exact
scoring pipeline (WoE-transform -> calibrated PD, and the scorecard points).

This mirrors backend.py so the validator scores borrowers the identical way the
deployed API does.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C

from dotenv import load_dotenv
load_dotenv(C.REPO_ROOT / ".env")   # pick up BASEL_DB_URL like the notebook does

def load_artifacts() -> dict:
    """Load the four joblib artifacts and pull features/avg_lgd out of meta."""
    missing = [str(p) for p in C.ARTIFACTS.values() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts (run basel_lite.ipynb top-to-bottom first):\n  "
            + "\n  ".join(missing)
        )
    art = {name: joblib.load(path) for name, path in C.ARTIFACTS.items()}
    meta = art["meta"]
    art["features"] = list(meta["features"])
    art["avg_lgd"] = float(meta["avg_lgd"])
    return art


def load_frame() -> pd.DataFrame:
    """
    Return the cleaned modeling frame (features + target).

    Priority: BASEL_VALIDATION_DATA file (parquet/csv) -> else the loans_clean
    table via BASEL_DB_URL (the same source the dashboard reads).
    """
    data_file = os.getenv(C.DATA_FILE_ENV, "").strip()
    if data_file:
        p = Path(data_file)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    db_url = os.getenv(C.DB_URL_ENV)
    if not db_url:
        raise RuntimeError(
            f"Set {C.DATA_FILE_ENV}=<file> for offline runs, or {C.DB_URL_ENV} "
            f"to read the `{C.TABLE}` table."
        )
    from sqlalchemy import create_engine
    engine = create_engine(db_url)
    return pd.read_sql(f"SELECT * FROM {C.TABLE}", engine)


def split(df: pd.DataFrame, features: list[str]):
    """Reproduce the notebook's train/test split exactly (cell 11)."""
    X = df[features]
    y = df[C.TARGET].astype(int)
    strat = y if C.SPLIT["stratify"] else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=C.SPLIT["test_size"],
        random_state=C.SPLIT["random_state"],
        stratify=strat,
    )
    return X_tr, X_te, y_tr.to_numpy(), y_te.to_numpy()


def pd_scores(art: dict, X: pd.DataFrame) -> np.ndarray:
    """Calibrated probability of default — identical to backend.py."""
    woe = art["binning"].transform(X[art["features"]], metric="woe")
    return art["pd_model"].predict_proba(woe)[:, 1]


def card_scores(art: dict, X: pd.DataFrame) -> np.ndarray:
    """300-850 scorecard points."""
    return np.asarray(art["scorecard"].score(X[art["features"]]))


def woe_frame(art: dict, X: pd.DataFrame) -> pd.DataFrame:
    """WoE-encoded features (used to fit the challenger model)."""
    return art["binning"].transform(X[art["features"]], metric="woe")
