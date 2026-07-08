"""ECL — multi-page PDF report (matplotlib PdfPages, no extra deps)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import config as C
from .engine import ECLResult, run

INK = "#0b1738"; GREEN = "#2e7d5b"; AMBER = "#c88a2e"; RED = "#b4413c"; BLUE = "#1f6feb"
STAGE_COLORS = {1: GREEN, 2: AMBER, 3: RED}


def _cover(pdf: PdfPages, res: ECLResult):
    p = res.portfolio
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.93, "Basel-Lite", ha="center", size=30, weight="bold", color=INK)
    fig.text(0.5, 0.895, "IFRS 9 — Expected Credit Loss", ha="center", size=14, color=INK)
    fig.text(0.5, 0.865, f"Probability-weighted, forward-looking  ·  {date.today().isoformat()}",
             ha="center", size=10, color="#555")

    # headline number
    fig.text(0.5, 0.80, f"{p['ecl_rate']:.2%}", ha="center", size=40, weight="bold", color=INK)
    fig.text(0.5, 0.765, "portfolio ECL coverage (ECL / EAD)", ha="center", size=10, color="#555")

    y = 0.71
    for label, val in [
        ("Loans", f"{p['n_loans']:,}"),
        ("Total EAD", f"{p['total_ead']:,.0f}"),
        ("Total ECL", f"{p['total_ecl']:,.0f}"),
        ("Average PD (TTC)", f"{p['avg_pd']:.1%}"),
        ("Average LGD", f"{p['avg_lgd']:.1%}"),
    ]:
        fig.text(0.12, y, label, size=10); fig.text(0.6, y, val, size=10, weight="bold")
        y -= 0.03

    # staging table
    y -= 0.02
    fig.add_artist(plt.Line2D([0.1, 0.9], [y, y], color="#ccc", transform=fig.transFigure)); y -= 0.03
    fig.text(0.1, y, "Staging (12-month vs lifetime ECL)", size=11, weight="bold"); y -= 0.03
    fig.text(0.12, y, "Stage", size=9, weight="bold"); fig.text(0.3, y, "Loans", size=9, weight="bold")
    fig.text(0.48, y, "Share", size=9, weight="bold"); fig.text(0.64, y, "ECL", size=9, weight="bold")
    fig.text(0.8, y, "Coverage", size=9, weight="bold"); y -= 0.028
    for s in (1, 2, 3):
        d = res.by_stage[f"stage_{s}"]
        horizon = "12-mo" if s == 1 else "lifetime"
        fig.text(0.12, y, f"{s} ({horizon})", size=9, color=STAGE_COLORS[s], weight="bold")
        fig.text(0.3, y, f"{d['count']:,}", size=9)
        fig.text(0.48, y, f"{d['share']:.1%}", size=9)
        fig.text(0.64, y, f"{d['ecl']:,.0f}", size=9)
        fig.text(0.8, y, f"{d['coverage']:.2%}", size=9)
        y -= 0.028

    # scenario table
    y -= 0.02
    fig.add_artist(plt.Line2D([0.1, 0.9], [y, y], color="#ccc", transform=fig.transFigure)); y -= 0.03
    fig.text(0.1, y, "Forward-looking scenarios (Vasicek overlay)", size=11, weight="bold"); y -= 0.03
    for name, _, _ in C.SCENARIOS:
        d = res.scenarios[name]
        fig.text(0.12, y, f"{name}  (w={d['weight']:.2f})", size=9)
        fig.text(0.6, y, f"ECL {d['total_ecl']:,.0f}", size=9); y -= 0.026

    fig.text(0.1, 0.05,
             "Staging is a PROXY: static LendingClub data lacks origination-vs-now PD history, "
             "so stages are assigned from absolute PD + delinquency, not a true SICR test. "
             "Macro Z-factors are stress assumptions, not fitted. See config.py.",
             size=7, color="#777", wrap=True)
    pdf.savefig(fig); plt.close(fig)


def _charts(pdf: PdfPages, res: ECLResult):
    fig, ax = plt.subplots(2, 2, figsize=(8.27, 11.69))

    # ECL by stage
    stages = [1, 2, 3]
    vals = [res.charts["stage_ecl"][s] for s in stages]
    ax[0, 0].bar([f"Stage {s}" for s in stages], vals, color=[STAGE_COLORS[s] for s in stages])
    ax[0, 0].set_title("ECL by stage"); ax[0, 0].set_ylabel("ECL")

    # ECL by scenario
    names = list(res.charts["scenario_ecl"].keys())
    svals = [res.charts["scenario_ecl"][k] for k in names]
    ax[0, 1].bar(names, svals, color=[BLUE, INK, RED][:len(names)])
    ax[0, 1].set_title("Total ECL by scenario"); ax[0, 1].set_ylabel("ECL")

    # default-timing weights (term structure shape)
    w = res.charts["weights"]
    ax[1, 0].bar(np.arange(1, len(w) + 1), w, color=INK)
    ax[1, 0].set_title("Default-timing weights (term structure)")
    ax[1, 0].set_xlabel("month"); ax[1, 0].set_ylabel("share of lifetime defaults")

    # coverage by grade
    if res.by_grade:
        grades = sorted(res.by_grade.keys())
        cov = [res.by_grade[g]["coverage"] for g in grades]
        ax[1, 1].bar(grades, cov, color=AMBER)
        ax[1, 1].set_title("ECL coverage by grade"); ax[1, 1].set_ylabel("ECL / EAD")
    else:
        ax[1, 1].axis("off")

    fig.suptitle("Basel-Lite — ECL evidence", size=13, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    pdf.savefig(fig); plt.close(fig)


def build_pdf(res: ECLResult | None = None, out_path: str | Path | None = None) -> Path:
    res = res or run()
    out_path = Path(out_path) if out_path else (C.REPO_ROOT / "ecl" / "report" / "basel_lite_ecl.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        _cover(pdf, res)
        _charts(pdf, res)
    return out_path


if __name__ == "__main__":
    print("Report ->", build_pdf())
