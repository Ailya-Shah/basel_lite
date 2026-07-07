"""
Generic model-validation metrics.

Nothing in this module knows anything about loans, Basel-Lite, or any specific
column. Every function takes plain arrays, so the same code validates any binary
classifier on any dataset. Everything project-specific lives in ``config.py``.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss


# --------------------------------------------------------------------------- #
# Discrimination — how well does the score rank-order good vs bad?
# --------------------------------------------------------------------------- #
def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_score))


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini coefficient = 2 * AUC - 1."""
    return float(2.0 * roc_auc_score(y_true, y_score) - 1.0)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max separation between the TPR and FPR curves."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def roc_points(y_true: np.ndarray, y_score: np.ndarray):
    """(fpr, tpr) arrays for plotting the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


# --------------------------------------------------------------------------- #
# Calibration — are the predicted probabilities trustworthy as probabilities?
# --------------------------------------------------------------------------- #
def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """
    Bin predictions into ``n_bins`` equal-frequency buckets and compare the mean
    predicted probability against the observed default rate in each bucket.

    Returns (mean_predicted, observed_rate, bin_counts) as arrays.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(y_prob, edges[1:-1])

    mean_pred, obs_rate, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        mean_pred.append(y_prob[m].mean())
        obs_rate.append(y_true[m].mean())
        counts.append(int(m.sum()))
    return np.array(mean_pred), np.array(obs_rate), np.array(counts)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Count-weighted mean absolute gap between predicted probability and observed
    rate across bins (ECE). 0 = perfectly calibrated.
    """
    mean_pred, obs_rate, counts = calibration_table(y_true, y_prob, n_bins)
    w = counts / counts.sum()
    return float(np.sum(w * np.abs(mean_pred - obs_rate)))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (mean squared error of the probabilities). Lower is better."""
    return float(brier_score_loss(y_true, y_prob))


# --------------------------------------------------------------------------- #
# Stability — has the score distribution shifted between two samples?
# --------------------------------------------------------------------------- #
def population_stability_index(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI between a baseline (``expected``) and a comparison (``actual``) sample.
    Bins are quantile edges of the baseline. Rule of thumb: <0.1 stable,
    0.1-0.25 minor shift, >0.25 material shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    e_prop = np.histogram(expected, bins=edges)[0] / len(expected)
    a_prop = np.histogram(actual, bins=edges)[0] / len(actual)

    eps = 1e-6  # avoid div-by-zero / log(0) in empty bins
    e_prop = np.clip(e_prop, eps, None)
    a_prop = np.clip(a_prop, eps, None)
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


# --------------------------------------------------------------------------- #
# Rank monotonicity — does default rate fall cleanly as the score rises?
# --------------------------------------------------------------------------- #
def default_rate_by_band(scores: np.ndarray, y_true: np.ndarray, n_bands: int = 10):
    """
    Split borrowers into ``n_bands`` score buckets (low score = risky) and return
    (band_midpoint, observed_default_rate, counts) per band, ordered low->high.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    edges = np.quantile(scores, np.linspace(0, 1, n_bands + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(scores, edges[1:-1])

    mids, rates, counts = [], [], []
    for b in range(n_bands):
        m = idx == b
        if m.sum() == 0:
            continue
        mids.append(scores[m].mean())
        rates.append(y_true[m].mean())
        counts.append(int(m.sum()))
    return np.array(mids), np.array(rates), np.array(counts)


def rank_monotonicity(scores: np.ndarray, y_true: np.ndarray, n_bands: int = 10) -> float:
    """
    Spearman correlation between score-band rank and observed default rate.
    A healthy scorecard is strongly negative (higher score -> lower default rate).
    """
    _, rates, _ = default_rate_by_band(scores, y_true, n_bands)
    rank = np.arange(len(rates))
    # Spearman = Pearson on ranks; rates already monotone-comparable by band index
    rr = np.argsort(np.argsort(rates))
    return float(np.corrcoef(rank, rr)[0, 1])
