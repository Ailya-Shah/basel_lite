"""
Everything tunable for the ECL engine lives here — and only here.

Mirrors validation/config.py: the maths modules stay generic, this file holds the
Basel-Lite specifics and the policy choices (staging thresholds, macro scenarios).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "assets" / "models"

ARTIFACTS = {
    "pd_model":  MODEL_DIR / "pd_model.joblib",
    "binning":   MODEL_DIR / "binning.joblib",
    "scorecard": MODEL_DIR / "scorecard.joblib",
    "meta":      MODEL_DIR / "model_meta.joblib",
}
# Optional: a saved Kaplan-Meier curve {"months": [...], "survival": [...]}.
# If absent, a parametric baseline shape is used (see data.load_survival_curve).
SURVIVAL_CURVE = MODEL_DIR / "survival_curve.joblib"

# --- data source (same as the dashboard / validator) ---
DB_URL_ENV    = "BASEL_DB_URL"
DATA_FILE_ENV = "BASEL_VALIDATION_DATA"     # reuse the validator's offline file
TABLE         = "loans_clean"
TARGET        = "default"
SAMPLE_N      = int(os.getenv("ECL_SAMPLE_N", "0")) or None  # None = whole book

# --- ECL mechanics ---
STAGE1_HORIZON_MONTHS = 12        # Stage 1 = 12-month ECL; Stage 2/3 = lifetime
DISCOUNT_WITH_EIR     = True      # discount at each loan's effective interest rate
FALLBACK_LGD          = 0.62      # used only if model_meta has no avg_lgd

# --- Staging (SICR) — HONEST PROXY -----------------------------------------
# Static LendingClub data has no origination-vs-now PD history, so true SICR
# can't be measured. We proxy it from observable risk at scoring time. This is
# the one part of the engine that is an approximation, and the report says so.
STAGING = {
    "stage3_pd":       0.50,   # >= this PD  -> treat as credit-impaired (Stage 3)
    "stage2_pd":       0.25,   # >= this PD  -> significant risk (Stage 2)
    "use_delinquency": True,   # delinq_2yrs >= 1 also pushes a loan to >= Stage 2
    "delinq_col":      "delinq_2yrs",
}

# --- Forward-looking macro overlay (Vasicek single-factor) -----------------
# TTC PD is shifted to point-in-time under each scenario factor Z, then the
# scenario ECLs are probability-weighted. Adverse = positive Z = higher PD.
ASSET_CORRELATION = 0.15          # rho (Basel retail-ish)
SCENARIOS = [                     # (name, Z, probability weight) — weights sum to 1
    ("upside",   -1.0, 0.25),
    ("baseline",  0.0, 0.50),
    ("adverse",  +2.0, 0.25),
]
