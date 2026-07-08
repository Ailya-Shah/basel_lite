"""
Acceptance criteria as tests. Each maps to a paragraph in the report.
The full validation runs once (session-scoped) and every check is asserted.
Run:  pytest validation/tests -v
"""
import pytest
from validation.validate import run


@pytest.fixture(scope="session")
def results():
    return run()


def test_no_leakage(results):
    assert results.get("leakage").passed, results.get("leakage").finding

def test_discrimination_gini(results):
    assert results.get("gini").passed, results.get("gini").finding

def test_discrimination_ks(results):
    assert results.get("ks").passed, results.get("ks").finding

def test_calibration(results):
    assert results.get("calibration").passed, results.get("calibration").finding

def test_score_stability_psi(results):
    assert results.get("psi").passed, results.get("psi").finding

def test_rank_monotonicity(results):
    assert results.get("monotonicity").passed, results.get("monotonicity").finding

def test_champion_beats_challenger(results):
    # A failure here is a legitimate finding (simpler model may suffice),
    # but for the shipped model we expect the GBM to hold its ground.
    assert results.get("challenger").passed, results.get("challenger").finding


def test_out_of_time_psi(results):
    if not any(c.name == "psi_out_of_time" for c in results.checks):
        pytest.skip("issue_d not persisted yet — run the notebook patch")
    assert results.get("psi_out_of_time").passed, results.get("psi_out_of_time").finding
