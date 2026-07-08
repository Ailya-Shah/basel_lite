"""ECL invariants as tests.  Run:  pytest ecl/tests -v"""
import numpy as np
import pytest
from ecl.engine import run
from ecl import term_structure as TS, ead as E, macro as MC


@pytest.fixture(scope="session")
def res():
    return run()


def test_scenario_weights_sum_to_one():
    from ecl import config as C
    assert abs(sum(w for _, _, w in C.SCENARIOS) - 1.0) < 1e-9


def test_ecl_never_exceeds_ead_times_lgd(res):
    pl = res.per_loan
    # a loan's ECL can't exceed its worst-case loss (EAD * LGD)
    assert (pl["ecl"] <= pl["ead"] * pl["lgd"] + 1e-6).all()


def test_ecl_nonnegative(res):
    assert (res.per_loan["ecl"] >= 0).all()


def test_adverse_worse_than_upside(res):
    # forward-looking overlay must make the adverse scenario cost more than upside
    s = res.scenarios
    assert s["adverse"]["total_ecl"] >= s["baseline"]["total_ecl"] >= s["upside"]["total_ecl"]


def test_term_structure_sums_to_pd():
    w = TS.survival_to_weights(np.arange(0, 37), np.linspace(1, 0.8, 37), 36)
    sched = TS.loan_pd_schedule(0.2, 36, w)
    assert abs(sched.sum() - 0.2) < 1e-9
    assert TS.pd_12m(sched) < 0.2


def test_ead_amortizes_down():
    ead = E.amortization_ead(10000, 12.0, 36)
    assert ead[0] > ead[-1] and (np.diff(ead) <= 1e-6).all()


def test_vasicek_monotonic_in_Z():
    pd0 = np.array([0.1])
    assert MC.vasicek_pit(pd0, +2, 0.15)[0] > MC.vasicek_pit(pd0, 0, 0.15)[0] > MC.vasicek_pit(pd0, -2, 0.15)[0]


def test_stage1_uses_12m_not_lifetime(res):
    # sanity: stage-1 coverage should be lighter than stage-2/3 (shorter horizon, lower PD)
    b = res.by_stage
    if b["stage_1"]["count"] and b["stage_2"]["count"]:
        assert b["stage_1"]["coverage"] <= b["stage_2"]["coverage"] + 1e-9
