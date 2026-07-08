"""
ECL — the engine.

Per loan, for each macro scenario:
    ECL_s = sum_{t=1..horizon} marginal_PD(t) * LGD * EAD(t) * DF(t)
where the horizon is 12 months (Stage 1) or the full term (Stage 2/3), marginal PD
comes from the term structure, EAD from the amortization schedule, and DF from the
effective interest rate. Scenario ECLs are then probability-weighted.

    ECL = sum_s weight_s * ECL_s
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import ead as E
from . import macro as MC
from . import staging as ST
from . import term_structure as TS


@dataclass
class ECLResult:
    per_loan: pd.DataFrame                 # one row per loan: pd, stage, ead, ecl, ...
    portfolio: dict = field(default_factory=dict)
    by_stage: dict = field(default_factory=dict)
    by_grade: dict = field(default_factory=dict)
    scenarios: dict = field(default_factory=dict)
    charts: dict = field(default_factory=dict)


def run() -> ECLResult:
    art = D.load_artifacts()
    df = D.load_frame().reset_index(drop=True)

    pd_ttc = D.pd_lifetime(art, df)
    terms = D.term_months(df)
    lgd = art["avg_lgd"]

    # baseline survival shape -> monthly default-timing weights (up to longest term)
    sm, sp = D.load_survival_curve()
    max_term = int(terms.max())
    weights_full = TS.survival_to_weights(sm, sp, horizon=max_term)

    # staging + scenarios
    stage = ST.assign_stages(pd_ttc, df, C.STAGING)
    scen_pd = MC.scenario_pds(pd_ttc, C.SCENARIOS, C.ASSET_CORRELATION)
    scen_w = MC.weights(C.SCENARIOS)

    loan_amnt = pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(0).to_numpy()
    int_rate = pd.to_numeric(df["int_rate"], errors="coerce").fillna(13.0).to_numpy()

    n = len(df)
    ecl_weighted = np.zeros(n)
    ecl_by_scen = {name: np.zeros(n) for name, _, _ in C.SCENARIOS}
    ead_at_orig = loan_amnt.copy()

    for i in range(n):
        term = int(terms[i])
        horizon = C.STAGE1_HORIZON_MONTHS if stage[i] == 1 else term
        horizon = min(horizon, term)
        w = weights_full[:term]
        ead_sched = E.amortization_ead(loan_amnt[i], int_rate[i], term)
        df_disc = E.discount_factors(int_rate[i], term, C.DISCOUNT_WITH_EIR)

        for name in ecl_by_scen:
            sched = TS.loan_pd_schedule(scen_pd[name][i], term, w)     # marginal PD/month
            contrib = sched[:horizon] * lgd * ead_sched[:horizon] * df_disc[:horizon]
            ecl_s = float(contrib.sum())
            ecl_by_scen[name][i] = ecl_s
            ecl_weighted[i] += scen_w[name] * ecl_s

    per_loan = pd.DataFrame({
        "pd_ttc": pd_ttc, "stage": stage, "term": terms,
        "ead": ead_at_orig, "lgd": lgd, "ecl": ecl_weighted,
        **{f"ecl_{name}": ecl_by_scen[name] for name in ecl_by_scen},
    })
    if "grade" in df.columns:
        per_loan["grade"] = df["grade"].values

    total_ead = float(ead_at_orig.sum())
    res = ECLResult(per_loan=per_loan)
    res.portfolio = {
        "n_loans": n,
        "total_ead": total_ead,
        "total_ecl": float(ecl_weighted.sum()),
        "ecl_rate": float(ecl_weighted.sum() / total_ead) if total_ead else 0.0,
        "avg_lgd": lgd,
        "avg_pd": float(pd_ttc.mean()),
    }
    res.scenarios = {name: {"total_ecl": float(ecl_by_scen[name].sum()),
                            "weight": scen_w[name]} for name in ecl_by_scen}
    # by stage
    res.by_stage = {}
    st_sum = ST.stage_summary(stage)
    for s in (1, 2, 3):
        m = stage == s
        res.by_stage[f"stage_{s}"] = {
            **st_sum[f"stage_{s}"],
            "ead": float(ead_at_orig[m].sum()),
            "ecl": float(ecl_weighted[m].sum()),
            "coverage": float(ecl_weighted[m].sum() / ead_at_orig[m].sum())
                        if ead_at_orig[m].sum() else 0.0,
        }
    # by grade
    if "grade" in per_loan.columns:
        g = per_loan.groupby("grade")
        res.by_grade = {str(k): {"ead": float(v["ead"].sum()),
                                 "ecl": float(v["ecl"].sum()),
                                 "coverage": float(v["ecl"].sum() / v["ead"].sum())
                                 if v["ead"].sum() else 0.0}
                        for k, v in g}
    res.charts = {
        "weights": weights_full,
        "stage_ecl": {s: res.by_stage[f"stage_{s}"]["ecl"] for s in (1, 2, 3)},
        "scenario_ecl": {name: ecl_by_scen[name].sum() for name in ecl_by_scen},
    }
    return res


def summary(res: ECLResult) -> str:
    p = res.portfolio
    lines = ["", "Basel-Lite — IFRS 9 Expected Credit Loss", "=" * 48,
             f"Loans            : {p['n_loans']:,}",
             f"Total EAD        : {p['total_ead']:,.0f}",
             f"Total ECL        : {p['total_ecl']:,.0f}",
             f"ECL / coverage   : {p['ecl_rate']:.2%} of exposure",
             "-" * 48, "By stage:"]
    for s in (1, 2, 3):
        d = res.by_stage[f"stage_{s}"]
        lines.append(f"  Stage {s}: {d['count']:>7,} loans "
                     f"({d['share']:5.1%})  coverage {d['coverage']:.2%}")
    lines.append("-" * 48)
    lines.append("Scenario ECL (probability-weighted into the total above):")
    for name, _, _ in C.SCENARIOS:
        d = res.scenarios[name]
        lines.append(f"  {name:<9} w={d['weight']:.2f}  ECL={d['total_ecl']:,.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary(run()))
