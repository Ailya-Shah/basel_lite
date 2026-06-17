"""
Basel-Lite — Credit Risk API (FastAPI backend)

Loads the models trained in basel_lite.ipynb and serves them:
  GET  /health        -> service + model status
  POST /score         -> score one borrower (PD, credit score, expected loss)
  POST /score_batch   -> score many borrowers + portfolio totals

Run locally:
  uvicorn app:app --reload
Then open the interactive docs at http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Load the artifacts saved by the notebook (Section 11)
# ---------------------------------------------------------------------------
MODEL_DIR = "assets/models"
pd_model  = joblib.load(f"{MODEL_DIR}/pd_model.joblib")   # calibrated PD model
binning   = joblib.load(f"{MODEL_DIR}/binning.joblib")    # WoE transformer
scorecard = joblib.load(f"{MODEL_DIR}/scorecard.joblib")  # interpretable points model
meta      = joblib.load(f"{MODEL_DIR}/model_meta.joblib")

FEATURES = meta["features"]
AVG_LGD  = float(meta["avg_lgd"])

app = FastAPI(title="Basel-Lite Credit Risk API", version="1.0")

# allow the Streamlit frontend (different port) to call this API
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Request schema — one borrower. Defaults double as the example in /docs.
# ---------------------------------------------------------------------------
class Borrower(BaseModel):
    loan_amnt: float = 15000
    term: str = " 36 months"
    int_rate: float = 13.0
    installment: float = 450.0
    grade: str = "C"
    sub_grade: str = "C2"
    emp_length: str = "5 years"
    home_ownership: str = "RENT"
    annual_inc: float = 65000
    verification_status: str = "Source Verified"
    purpose: str = "debt_consolidation"
    dti: float = 18.0
    delinq_2yrs: float = 0
    inq_last_6mths: float = 1
    open_acc: float = 10
    pub_rec: float = 0
    revol_bal: float = 12000
    revol_util: float = 45.0
    total_acc: float = 25
    application_type: str = "Individual"
    mort_acc: float = 1
    pub_rec_bankruptcies: float = 0
    addr_state: str = "CA"
    fico_score: float = 690


def _risk_band(pd_hat: float) -> str:
    return "Low" if pd_hat < 0.10 else "Medium" if pd_hat < 0.25 else "High"


def _score_frame(df: pd.DataFrame):
    """Run the full pipeline on a DataFrame of borrowers."""
    df = df[FEATURES]                              # enforce column order
    woe = binning.transform(df, metric="woe")      # WoE-encode for the PD model
    pds = pd_model.predict_proba(woe)[:, 1]
    scores = scorecard.score(df)
    eads = df["loan_amnt"].to_numpy()
    els = pds * AVG_LGD * eads
    return pds, scores, eads, els


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "avg_lgd": round(AVG_LGD, 4), "n_features": len(FEATURES)}


@app.post("/score")
def score(borrower: Borrower):
    df = pd.DataFrame([borrower.model_dump()])
    pds, scores, eads, els = _score_frame(df)
    pd_hat = float(pds[0])
    return {
        "probability_of_default": round(pd_hat, 4),
        "credit_score": round(float(scores[0])),
        "risk_band": _risk_band(pd_hat),
        "lgd": round(AVG_LGD, 4),
        "ead": round(float(eads[0]), 2),
        "expected_loss": round(float(els[0]), 2),
    }


@app.post("/score_batch")
def score_batch(borrowers: list[Borrower]):
    df = pd.DataFrame([b.model_dump() for b in borrowers])
    pds, scores, eads, els = _score_frame(df)
    return {
        "n": len(borrowers),
        "total_ead": round(float(eads.sum()), 2),
        "total_expected_loss": round(float(els.sum()), 2),
        "expected_loss_rate": round(float(els.sum() / eads.sum()), 4),
        "results": [
            {"pd": round(float(p), 4), "credit_score": round(float(s)),
             "expected_loss": round(float(e), 2)}
            for p, s, e in zip(pds, scores, els)
        ],
    }