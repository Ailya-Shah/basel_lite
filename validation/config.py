"""
Everything Basel-Lite-specific lives here — and ONLY here.

To point this validator at a different model, you copy the ``validation/`` folder
out and edit this one file. The metric functions in ``metrics.py`` never change.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Where the trained artifacts live (saved by basel_lite.ipynb, Section 11)
# --------------------------------------------------------------------------- #
# Repo root = one level up from this file's folder.
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "assets" / "models"

ARTIFACTS = {
    "pd_model":  MODEL_DIR / "pd_model.joblib",    # calibrated PD model (CalibratedClassifierCV)
    "binning":   MODEL_DIR / "binning.joblib",     # optbinning WoE transformer
    "scorecard": MODEL_DIR / "scorecard.joblib",   # optbinning Scorecard (300-850)
    "meta":      MODEL_DIR / "model_meta.joblib",  # {features, categorical, avg_lgd}
}

# --------------------------------------------------------------------------- #
# Where the holdout data comes from
# --------------------------------------------------------------------------- #
# Default: read the same `loans_clean` table the dashboard uses, from BASEL_DB_URL.
# Override with BASEL_VALIDATION_DATA=<path.parquet|.csv> to validate offline
# (handy for CI, where there's no MySQL).
DB_URL_ENV      = "BASEL_DB_URL"
DATA_FILE_ENV   = "BASEL_VALIDATION_DATA"
TABLE           = "loans_clean"
TARGET          = "default"

# Reproduce the notebook's exact split (cell 11) so the holdout is identical.
SPLIT = {"test_size": 0.25, "random_state": 42, "stratify": True}

# The 300-850 scorecard range (informational, used for labelling).
SCORE_RANGE = (300, 850)

# --------------------------------------------------------------------------- #
# Leakage blocklist — fields known only AFTER origination.
# The check asserts none of these leaked into the model's feature set.
# --------------------------------------------------------------------------- #
POST_ORIGINATION_FIELDS = {
    "recoveries", "collection_recovery_fee", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "out_prncp", "out_prncp_inv",
    "funded_amnt", "funded_amnt_inv", "loan_status",
}

# --------------------------------------------------------------------------- #
# Acceptance criteria. These are the pass/fail bars the tests assert against.
# They're deliberately explicit and tunable — a real MRM team writes these down.
# A FAILING check is not a crash; it's a documented finding in the report.
# --------------------------------------------------------------------------- #
THRESHOLDS = {
    "gini_min":        0.30,   # minimum rank-ordering power
    "ks_min":          0.25,   # minimum KS separation
    "ece_max":         0.05,   # max expected calibration error
    "psi_max":         0.25,   # max score-distribution shift (train vs holdout)
    "monotonic_max":  -0.80,   # Spearman(band, default rate) must be <= this
    "challenger_tol":  0.02,   # champion Gini must be >= challenger Gini - tol
}

# Toggle: the DB read can be slow; cache the split to parquet after first run.
CACHE_SPLIT = os.getenv("BASEL_VALIDATION_CACHE", "").strip() or None
