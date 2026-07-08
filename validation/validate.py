"""
Run the full validation and return a structured results object.

Each check produces: value, threshold, pass/fail, and a one-line finding.
The results feed both the PDF report (report.py) and the tests (test_validation.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config as C
from . import data as D
from . import metrics as M


@dataclass
class Check:
    name: str
    value: float
    threshold: float
    passed: bool
    finding: str


@dataclass
class Results:
    checks: list[Check] = field(default_factory=list)
    charts: dict = field(default_factory=dict)   # arrays for plotting
    info: dict = field(default_factory=dict)      # extra headline numbers

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def get(self, name: str) -> Check:
        return next(c for c in self.checks if c.name == name)

    def as_dict(self) -> dict:
        return {"checks": [asdict(c) for c in self.checks], "info": self.info}


def run() -> Results:
    art = D.load_artifacts()
    df = D.load_frame()
    X_tr, X_te, y_tr, y_te = D.split(df, art["features"])

    # ---- score the holdout -------------------------------------------------
    pd_te = D.pd_scores(art, X_te)
    pd_tr = D.pd_scores(art, X_tr)
    sc_te = D.card_scores(art, X_te)
    sc_tr = D.card_scores(art, X_tr)

    res = Results()
    T = C.THRESHOLDS

    # ---- 1. leakage --------------------------------------------------------
    leaked = sorted(set(art["features"]) & C.POST_ORIGINATION_FIELDS)
    res.checks.append(Check(
        "leakage", value=len(leaked), threshold=0, passed=len(leaked) == 0,
        finding=("No post-origination fields in the feature set."
                 if not leaked else f"LEAKAGE: {leaked}"),
    ))

    # ---- 2. discrimination -------------------------------------------------
    g = M.gini(y_te, pd_te)
    ks = M.ks_statistic(y_te, pd_te)
    res.checks.append(Check(
        "gini", g, T["gini_min"], g >= T["gini_min"],
        f"Gini {g:.3f} on the holdout (AUC {M.auc(y_te, pd_te):.3f}).",
    ))
    res.checks.append(Check(
        "ks", ks, T["ks_min"], ks >= T["ks_min"],
        f"KS separation {ks:.3f}.",
    ))

    # ---- 3. calibration ----------------------------------------------------
    ece = M.expected_calibration_error(y_te, pd_te)
    res.checks.append(Check(
        "calibration", ece, T["ece_max"], ece <= T["ece_max"],
        f"Expected calibration error {ece:.3f} (Brier {M.brier(y_te, pd_te):.3f}).",
    ))

    # ---- 4. stability (train vs holdout) -----------------------------------
    psi = M.population_stability_index(sc_tr, sc_te)
    res.checks.append(Check(
        "psi", psi, T["psi_max"], psi <= T["psi_max"],
        f"Score PSI train->holdout {psi:.3f}. "
        f"(Out-of-time PSI needs issue_d persisted; see report notes.)",
    ))

    # ---- 5. rank monotonicity ---------------------------------------------
    mono = M.rank_monotonicity(sc_te, y_te)
    res.checks.append(Check(
        "monotonicity", mono, T["monotonic_max"], mono <= T["monotonic_max"],
        f"Spearman(score band, default rate) = {mono:.3f} (want strongly negative).",
    ))

    # ---- 6. champion vs challenger ----------------------------------------
    woe_tr, woe_te = D.woe_frame(art, X_tr), D.woe_frame(art, X_te)
    challenger = LogisticRegression(max_iter=1000).fit(woe_tr, y_tr)
    ch_pd = challenger.predict_proba(woe_te)[:, 1]
    ch_gini = M.gini(y_te, ch_pd)
    passed = g >= ch_gini - T["challenger_tol"]
    res.checks.append(Check(
        "challenger", g - ch_gini, -T["challenger_tol"], passed,
        f"Champion Gini {g:.3f} vs logistic challenger {ch_gini:.3f} "
        f"(delta {g - ch_gini:+.3f}).",
    ))

    # ---- charts + headline info -------------------------------------------
    fpr, tpr = M.roc_points(y_te, pd_te)
    cal_pred, cal_obs, _ = M.calibration_table(y_te, pd_te)
    band_mid, band_rate, _ = M.default_rate_by_band(sc_te, y_te)
    res.charts = {
        "roc": (fpr, tpr, M.auc(y_te, pd_te)),
        "calibration": (cal_pred, cal_obs),
        "score_hist_train": sc_tr, "score_hist_test": sc_te,
        "band": (band_mid, band_rate),
        "champion_challenger": (g, ch_gini),
    }
    ead = X_te["loan_amnt"].to_numpy()
    el = pd_te * art["avg_lgd"] * ead
    res.info = {
        "n_train": len(y_tr), "n_holdout": len(y_te),
        "holdout_default_rate": float(y_te.mean()),
        "avg_lgd": art["avg_lgd"],
        "expected_loss_rate": float(el.sum() / ead.sum()),
        "avg_pd": float(pd_te.mean()),
    }

    # ---- 4b. out-of-time stability (vintage PSI) --------------------------
    if C.ISSUE_DATE_COL in df.columns:
        yr = pd.to_datetime(df[C.ISSUE_DATE_COL], errors="coerce").dt.year.to_numpy()
        all_sc = D.card_scores(art, df)                       # score the full book
        (b0, b1), (c0, c1) = C.VINTAGE_BASELINE, C.VINTAGE_COMPARE
        base = all_sc[(yr >= b0) & (yr <= b1)]
        comp = all_sc[(yr >= c0) & (yr <= c1)]
        if len(base) >= 100 and len(comp) >= 100:
            psi_oot = M.population_stability_index(base, comp)
            res.checks.append(Check(
                "psi_out_of_time", psi_oot, T["psi_oot_max"], psi_oot <= T["psi_oot_max"],
                f"Score PSI {b0}-{b1} vs {c0}-{c1}: {psi_oot:.3f} "
                f"(n {len(base):,} vs {len(comp):,}).",
            ))
            res.charts["psi_oot"] = (base, comp, (b0, b1, c0, c1))
        else:
            res.checks.append(Check(
                "psi_out_of_time", float("nan"), T["psi_oot_max"], True,
                "Skipped: a vintage window has too few completed loans.",
            ))
    else:
        res.info["psi_oot_note"] = ("issue_d not in loans_clean — apply the notebook "
                                    "patch and re-run to enable out-of-time PSI.")

    return res


def _summary(res: Results) -> str:
    lines = ["", "Basel-Lite — validation summary", "=" * 46]
    for c in res.checks:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"[{mark}] {c.name:<14} {c.finding}")
    lines.append("-" * 46)
    lines.append("OVERALL: " + ("ALL CHECKS PASSED" if res.all_passed
                                 else "FINDINGS PRESENT — see report"))
    return "\n".join(lines)


if __name__ == "__main__":
    print(_summary(run()))
