"""
Render a Results object into a multi-page PDF validation report.

Uses matplotlib's PdfPages only (already a Basel-Lite dependency) — no extra
libraries. One command regenerates the whole thing deterministically.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import config as C
from .validate import Results, run

INK = "#0b1738"
GREEN = "#2e7d5b"
RED = "#b4413c"
BLUE = "#1f6feb"


def _cover_page(pdf: PdfPages, res: Results):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.5, 0.92, "Basel-Lite", ha="center", size=30, weight="bold", color=INK)
    fig.text(0.5, 0.885, "Independent Model Validation Report", ha="center",
             size=14, color=INK)
    fig.text(0.5, 0.855, f"Model: calibrated LightGBM PD  ·  {date.today().isoformat()}",
             ha="center", size=10, color="#555")

    # verdict banner
    ok = res.all_passed
    fig.patches.append(plt.Rectangle((0.1, 0.79), 0.8, 0.03,
                       transform=fig.transFigure,
                       color=GREEN if ok else RED, alpha=0.85))
    fig.text(0.5, 0.796, "ALL CHECKS PASSED" if ok else "FINDINGS PRESENT",
             ha="center", size=12, weight="bold", color="white")

    # checks table
    y = 0.74
    fig.text(0.1, y, "Check", size=10, weight="bold")
    fig.text(0.42, y, "Result", size=10, weight="bold")
    fig.text(0.86, y, "Status", size=10, weight="bold")
    y -= 0.02
    fig.add_artist(plt.Line2D([0.1, 0.9], [y, y], color="#ccc", transform=fig.transFigure))
    y -= 0.028
    for c in res.checks:
        finding = c.finding if len(c.finding) <= 66 else c.finding[:63] + "..."
        fig.text(0.1, y, c.name.replace("_", " ").title(), size=9)
        fig.text(0.42, y, finding, size=7.5, color="#333")
        fig.text(0.86, y, "PASS" if c.passed else "FAIL", size=9, weight="bold",
                 color=GREEN if c.passed else RED)
        y -= 0.045

    # headline portfolio numbers
    i = res.info
    y -= 0.01
    fig.add_artist(plt.Line2D([0.1, 0.9], [y, y], color="#ccc", transform=fig.transFigure))
    y -= 0.03
    fig.text(0.1, y, "Portfolio snapshot (holdout)", size=10, weight="bold"); y -= 0.028
    for label, val in [
        ("Holdout loans", f"{i['n_holdout']:,}"),
        ("Observed default rate", f"{i['holdout_default_rate']:.1%}"),
        ("Average PD", f"{i['avg_pd']:.1%}"),
        ("Average LGD (portfolio)", f"{i['avg_lgd']:.1%}"),
        ("Expected Loss rate", f"{i['expected_loss_rate']:.1%}"),
    ]:
        fig.text(0.12, y, label, size=9); fig.text(0.6, y, val, size=9, weight="bold")
        y -= 0.026

    fig.text(0.1, 0.06,
             "Acceptance thresholds are explicit and tunable in validation/config.py. "
             "A failing check is a documented finding, not an error. Numbers regenerate "
             "deterministically from the saved model + fixed-seed holdout.",
             size=7, color="#777", wrap=True)
    pdf.savefig(fig); plt.close(fig)


def _charts_page(pdf: PdfPages, res: Results):
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
    ch = res.charts

    fpr, tpr, a = ch["roc"]
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color=BLUE, label=f"PD model (AUC {a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_title("ROC — discrimination"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8)

    cp, co = ch["calibration"]
    ax = axes[0, 1]
    ax.plot(cp, co, "o-", color=RED, label="model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set_title("Calibration"); ax.set_xlabel("Predicted PD"); ax.set_ylabel("Observed rate")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(ch["score_hist_train"], bins=40, alpha=0.55, label="train", color="#7b86bb")
    ax.hist(ch["score_hist_test"], bins=40, alpha=0.55, label="holdout", color=GREEN)
    ax.set_title(f"Score stability (PSI {res.get('psi').value:.3f})")
    ax.set_xlabel("Score (300-850)"); ax.set_ylabel("Borrowers"); ax.legend(fontsize=8)

    bm, br = ch["band"]
    ax = axes[1, 1]
    ax.plot(bm, br, "o-", color=INK)
    ax.set_title("Default rate by score band"); ax.set_xlabel("Score band (mean)")
    ax.set_ylabel("Observed default rate")

    fig.suptitle("Basel-Lite — validation evidence", size=13, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    pdf.savefig(fig); plt.close(fig)


def build_pdf(res: Results | None = None, out_path: str | Path | None = None) -> Path:
    res = res or run()
    out_path = Path(out_path) if out_path else (C.REPO_ROOT / "validation" / "report"
                                                / "basel_lite_validation.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        _cover_page(pdf, res)
        _charts_page(pdf, res)
    return out_path


if __name__ == "__main__":
    p = build_pdf()
    print(f"Report written to {p}")
