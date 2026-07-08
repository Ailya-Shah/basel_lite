# Basel-Lite — Credit Risk & Capital Engine

> An end-to-end credit risk system on real LendingClub data: it estimates a borrower's **probability of default**, assigns a **300–850 credit score**, measures **loss given default** from real recoveries, and rolls everything up into a portfolio **Expected Loss**. On top of the model sit two second-line layers a real risk desk would demand: an independent **model-validation** pipeline and a forward-looking **IFRS 9 Expected Credit Loss** engine. Built as a full stack — a modeling notebook, a FastAPI service, and a live Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-PD%20model-2DD4BF)
![MySQL](https://img.shields.io/badge/MySQL-data-4479A1?logo=mysql&logoColor=white)
![Validation](https://img.shields.io/badge/validation-8%2F8%20checks%20passing-2e7d5b)
![IFRS 9](https://img.shields.io/badge/IFRS%209-ECL%20engine-6c5ce7)

**Report authors:** Ailya Shah  

**Repository owner / author note:** Ailya Shah, Data Science at SEECS
##
## Abstract
Banks lose money when borrowers default, and the hard part isn't lending — it's pricing that risk before it happens. Basel-Lite is an end-to-end credit risk system that does exactly that, built on a 200,000-loan sample of the 2007–2018 LendingClub book (≈119,000 *completed* loans used for modeling, after dropping still-current ones). It estimates each borrower's probability of default with a calibrated LightGBM model, assigns a 300–850 credit score using a Weight-of-Evidence scorecard like the ones banks actually deploy, and measures loss given default straight from real recovery data. Those pieces combine into the Basel formula — Expected Loss = PD × LGD × EAD — to compute the capital a lender should reserve, loan by loan and across a sampled book. Every model uses only information available at application time, so there's no data leakage inflating the results. The whole thing ships as a real stack: a MySQL data layer, a FastAPI scoring service, and a live Streamlit dashboard where dragging a borrower's FICO score watches their default risk and expected loss recalculate in real time. SHAP explains every prediction, calibration curves prove the probabilities are trustworthy, and survival analysis models when defaults strike, not just whether. On top of the model sit two second-line layers: an **independent validation pipeline** that re-scores a holdout and runs eight acceptance checks (discrimination, calibration, stability, out-of-time drift, rank monotonicity, leakage, and a champion-vs-challenger benchmark), and an **IFRS 9 Expected Credit Loss engine** that turns the single PD into a forward-looking, staged, scenario-weighted loss allowance. It's not a notebook — it's a deployable risk engine, shipped with the checks and provisioning a real risk desk would demand.

##
![Borrower scorer](app-ss/app.png)
##

---

## What it does

- **Probability of Default (PD)** — a calibrated LightGBM model trained on application-time features only (no leakage).
- **Credit scorecard** — a Weight-of-Evidence + logistic scorecard scaled to a 300–850 range, the way regulated credit scores are actually built.
- **Loss Given Default (LGD)** — measured empirically from real recovery data on charged-off loans.
- **Expected Loss** — `EL = PD × LGD × EAD`, computed per loan and aggregated to a portfolio number.
- **Timing of default** — survival analysis (Kaplan–Meier + Cox) for *when* defaults happen, on top of *whether* they happen.
- **Independent validation** — a `validation/` package that re-scores a holdout and runs eight acceptance checks as a `pytest` suite, then renders a PDF validation report.
- **IFRS 9 ECL engine** — an `ecl/` package that stages each loan (12-month vs lifetime), builds a PD term structure from the survival curve, amortizes exposure, and probability-weights macro scenarios into a forward-looking loss allowance, with its own PDF report and tests.
- **Live app** — a borrower scorer that updates as you drag the sliders, plus a portfolio dashboard that values a sampled slice of the book in one click.

---

## The app

### Live borrower scorer
Set a borrower's profile and the assessment updates instantly — probability of default, a credit-score gauge that fills green → amber → red, the risk band, and the expected loss on that loan, broken down into `PD × LGD × EAD`.

![Borrower scorer](app-ss/front.png)

Every input is adjustable, with the finer credit-history fields tucked into an expander:

![Advanced inputs](app-ss/advanced-options.png)

### Portfolio risk
Sample any number of loans from the book and value them in one pass — total exposure, total expected loss, loss rate, average PD, and the loss broken down by credit grade.

![Portfolio risk](app-ss/portfolio-risk.png)

---

## How it works

### Data & leakage control
The model is trained on a sample of the **LendingClub 2007–2018** loan book. The target is built from loan status — *Charged Off* = default (1), *Fully Paid* = good (0), with in-progress loans dropped. Crucially, only features **known at application time** are used; post-origination fields (`recoveries`, `total_pymnt`, last-pull FICO, etc.) are excluded so the model can't "cheat" by seeing the outcome.

### Feature strength — Information Value
Every feature is binned and scored by Information Value, the standard way a credit team ranks predictors before building a scorecard.

### Probability of Default — model performance
A calibrated LightGBM classifier, evaluated with the metrics banks actually use (AUC, KS, Gini), and then calibrated so its outputs are true probabilities — essential for the expected-loss math.

![ROC curve](assets/images/roc_curve.png)
![Calibration curve](assets/images/calibration_curve.png)

### Explainability — SHAP
Which features drive each prediction, and in which direction.

![SHAP feature importance](assets/images/shap_importance.png)

### Loss Given Default
Measured from actual recoveries on charged-off loans, not assumed.

![LGD distribution](assets/images/lgd_distribution.png)

### When defaults happen — survival analysis
A Kaplan–Meier curve for how loan survival decays over time. This same curve drives the PD term structure inside the ECL engine.

![Survival curve](assets/images/survival_curve.png)

### Exploratory findings
Default rate climbs cleanly across LendingClub's own risk grades — a sanity check that the target is correct.

![Default rate by grade](assets/images/default_rate_by_grade.png)
![Credit score distribution](assets/images/score_distribution.png)

---

## Results

*Numbers below are the exact values printed by the notebook on the current 200K sample; they will shift slightly if you resample.*

| Metric | Value |
|---|---|
| Modeling sample (completed loans) | 119,060 |
| Portfolio default rate | 19.8% |
| Average recovery rate (measured) | 37.8% |
| Average LGD (measured) | 62.2% |
| PD model — ROC AUC | 0.713 |
| PD model — KS | 0.312 |
| PD model — Gini | 0.427 |
| Test-set average PD | 19.9% |
| Test-set Expected Loss rate | 13.3% of exposure |

---

## Model validation

Building a model is the first line; **checking it is the second.** The `validation/`
package acts as an independent reviewer: it loads the saved artifacts, re-scores a
fixed-seed holdout the exact way the API does, and runs eight acceptance checks —
encoded as a `pytest` suite and rendered into a PDF report. Every check has an
explicit, tunable threshold in `validation/config.py`; a *failing* check is a
documented finding, not a crash.

**Latest run — all checks passing on the LendingClub holdout:**

| Check | Result | Status |
|---|---|---|
| Leakage | No post-origination fields in the feature set | ✅ Pass |
| Discrimination — Gini | 0.426 (AUC 0.713) | ✅ Pass |
| Discrimination — KS | 0.308 | ✅ Pass |
| Calibration | Expected calibration error 0.009 (Brier 0.144) | ✅ Pass |
| Stability — PSI (in-sample) | 0.000 (train → holdout) | ✅ Pass |
| Stability — PSI (out-of-time) | 0.008 (2013–14 vs 2015–16, n 31,613 vs 59,193) | ✅ Pass |
| Rank monotonicity | Spearman −1.000 (score band vs default rate) | ✅ Pass |
| Champion vs challenger | Gini 0.426 vs logistic 0.419 (Δ +0.007) | ✅ Pass |

📄 Full report: [`validation/report/basel_lite_validation.pdf`](validation/report/basel_lite_validation.pdf)

**What the results mean.** The calibrated PDs sit within roughly one percentage
point of observed default rates across deciles (ECE 0.009), which is precisely what
licenses feeding them straight into `EL = PD × LGD × EAD` — an uncalibrated ranker
would distort the capital number even with the same AUC. Rank ordering is clean
(Spearman −1.000): default rate falls monotonically across every score band with no
reversals. The out-of-time PSI (0.008) shows the score distribution is stable across
loan vintages two-plus years apart — the "is my model still valid later" check that
in-sample stability can't answer.

The champion-vs-challenger check is the honest one. The LightGBM beats a plain
WoE-logistic regression by only **0.007 Gini** — a documented finding, not a
failure. It says most of the signal is already captured linearly through the WoE
transform, so the simpler, more interpretable logistic model is a defensible
production choice; the GBM is retained here for the marginal lift and the
explainability tooling around it.

**Run it:**

```bash
python -m validation.run               # prints the pass/fail summary + writes the PDF
python -m pytest validation/tests -v   # runs the eight checks as tests
```

The validator reads the same `loans_clean` table the dashboard uses (via
`BASEL_DB_URL`); set `BASEL_VALIDATION_DATA=<file.parquet>` to run offline / in CI.
Out-of-time PSI compares two matured vintages (2013–14 vs 2015–16); recent vintages
are less matured in the 2018 snapshot, so those windows are chosen deliberately.

---

## IFRS 9 — Expected Credit Loss

A single PD tells you *whether* a loan defaults; provisioning under **IFRS 9** asks a
harder question — how much loss to reserve today, looking forward, over the right
horizon. The `ecl/` engine answers it. For each loan and each macro scenario it
computes

```
ECL = Σ over months  [ marginal PD(t) × LGD × EAD(t) × discount(t) ]
```

then probability-weights the scenarios. It reuses everything upstream: the survival
curve supplies the **PD term structure** (a monthly marginal PD that sums back to the
loan's lifetime PD), an **amortization schedule** gives a declining EAD instead of a
flat loan amount, the recovery-based **LGD** carries over, and a **Vasicek single-factor
overlay** shifts through-the-cycle PD to point-in-time under upside / baseline / adverse
scenarios.

**Latest run — full completed book (119,060 loans):**

| Portfolio | Value |
|---|---|
| Total EAD | $1.72bn |
| Total ECL (probability-weighted) | $98.5m |
| **ECL coverage (ECL / EAD)** | **5.75%** |

| Stage | Horizon | Loans | Share | Coverage (ECL / EAD) |
|---|---|---|---|---|
| Stage 1 | 12-month | 68,806 | 57.8% | 2.27% |
| Stage 2 | lifetime | 47,560 | 39.9% | 9.35% |
| Stage 3 | lifetime | 2,694 | 2.3% | 16.12% |

| Scenario | Weight | Total ECL |
|---|---|---|
| Upside | 0.25 | $47.3m |
| Baseline | 0.50 | $82.5m |
| Adverse | 0.25 | $181.9m |

📄 Full report: [`ecl/report/basel_lite_ecl.pdf`](ecl/report/basel_lite_ecl.pdf)

**What the results mean.** Portfolio coverage of **5.75%** sits well below the 19.8%
lifetime default rate — as it should: most loans are Stage 1 (only 12 months of risk
counted) and exposure amortizes down over the life, so lifetime default risk lands
after much of the balance is already repaid. Coverage rises cleanly by stage
(2.27% → 9.35% → 16.12%): worse stages cost more per dollar of exposure, which is the
key internal-consistency check. The scenario spread (upside $47m < baseline $83m <
adverse $182m) shows the forward-looking overlay responding sensibly to stress, with
the adverse case ≈2.2× baseline.

**Run it:**

```bash
python -m ecl.run                 # prints the ECL summary + writes the PDF
python -m pytest ecl/tests -v     # eight invariant checks (schedule sums to PD, ECL <= EAD*LGD, adverse >= baseline >= upside, ...)
```

> **Honest limitations (also printed on the report).** **Staging is a proxy** — static
> LendingClub data has no origination-vs-now PD history, so a true "significant increase
> in credit risk" (SICR) test isn't possible; stages are assigned from absolute PD +
> delinquency (thresholds in `ecl/config.py`). The **macro Z-factors are stress
> assumptions, not fitted** — the data-driven upgrade is to regress vintage default
> rates on a real macro series (e.g. FRED unemployment) using `issue_d`. Default
> *timing* comes from the saved Kaplan–Meier curve (`assets/models/survival_curve.joblib`);
> a parametric shape is used if that artifact is absent.

---

## Architecture

```mermaid
flowchart LR
    L[(MySQL<br/>loans — raw sample)] --> B[basel_lite.ipynb<br/>clean · train · save]
    B --> Lc[(MySQL<br/>loans_clean)]
    B --> C[/assets/models<br/>*.joblib/]
    C --> D[backend.py<br/>FastAPI]
    Lc --> E[frontend.py<br/>Streamlit dashboard]
    C --> V[validation/<br/>checks + PDF report]
    Lc --> V
    C --> X[ecl/<br/>IFRS 9 ECL + PDF report]
    Lc --> X
    D -->|/score · /score_batch| E
    U((User)) --> E
```

The notebook cleans the raw `loans` sample, writes `loans_clean`, and saves the trained models; FastAPI loads the models and serves predictions; Streamlit reads `loans_clean` for the portfolio view and calls the API for scoring; the validation and ECL layers read the same artifacts and data to produce their reports.

---

## Project structure

```
basel_lite/
├── basel_lite.ipynb        # full modeling pipeline: clean → EDA → scorecard → PD/LGD → EL → survival
├── backend.py              # FastAPI service: /score, /score_batch, /health
├── frontend.py             # Streamlit dashboard: live scorer + portfolio view
├── validation/             # independent model-validation layer
│   ├── metrics.py          # generic checks: Gini, KS, PSI, calibration, monotonicity
│   ├── config.py           # Basel-Lite specifics: paths, split, leakage list, thresholds
│   ├── data.py             # loads artifacts + holdout, runs the scoring pipeline
│   ├── validate.py         # runs all checks → results
│   ├── report.py           # results → PDF report
│   ├── run.py              # validate + report in one command
│   ├── report/             # generated: basel_lite_validation.pdf
│   └── tests/              # pytest acceptance criteria (one per check)
├── ecl/                    # IFRS 9 Expected Credit Loss engine
│   ├── term_structure.py   # survival timing → marginal PD per month
│   ├── ead.py              # amortization schedule + discount factors
│   ├── staging.py          # Stage 1/2/3 (SICR proxy)
│   ├── macro.py            # Vasicek PIT transform + scenario weighting
│   ├── engine.py           # assembles per-loan ECL + portfolio aggregation
│   ├── data.py             # loads artifacts, scores PD, loads survival curve
│   ├── config.py           # scenarios, staging thresholds, LGD, paths
│   ├── report.py           # results → PDF report
│   ├── run.py              # compute + report in one command
│   ├── report/             # generated: basel_lite_ecl.pdf
│   └── tests/              # pytest invariant checks
├── .streamlit/
│   └── config.toml         # dark theme
├── assets/
│   ├── images/             # charts, generated by the notebook (Section 12)
│   └── models/             # trained artifacts: pd_model, binning, scorecard, model_meta, survival_curve
├── app-ss/                 # app screenshots (used in this README)
├── .env.example            # template for BASEL_DB_URL / BASEL_API_URL (copy to .env)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting started

### Prerequisites
- Python 3.11+
- MySQL (with a schema named `basel_lite`)

### 1. Get the data
Download the LendingClub accepted-loans file from Kaggle
(`wordsforthewise/lending-club`) and load a sample into a `loans` table in MySQL.
Running the notebook's cleaning cells then writes the modeling-ready `loans_clean`
table (the dashboard's Portfolio page reads from it). The raw data is **not**
committed to this repo.

### 2. Install dependencies & configure secrets
```bash
python -m venv .venv
# activate it, then:
pip install -r requirements.txt

# copy the env template and fill in your MySQL URL
cp .env.example .env        # Windows: copy .env.example .env
```
`.env` holds `BASEL_DB_URL` (and optionally `BASEL_API_URL`); it is gitignored, so
no credentials live in the repo.

### 3. Train the models
Open `basel_lite.ipynb`, select the `.venv` kernel, and run all cells top to bottom.
This saves the trained artifacts into `assets/models/` (including `survival_curve.joblib`
for the ECL engine) and the charts into `assets/images/`.

### 4. Start the backend
```bash
uvicorn backend:app --reload
```
Interactive API docs at <http://127.0.0.1:8000/docs>.

### 5. Start the frontend (new terminal)
```bash
streamlit run frontend.py
```
Opens at <http://localhost:8501>.

### 6. Validate the model & compute ECL (optional, any time)
```bash
python -m validation.run    # → validation/report/basel_lite_validation.pdf
python -m ecl.run           # → ecl/report/basel_lite_ecl.pdf
```

---

## Tech stack

| Layer | Tool |
|---|---|
| Data store | MySQL |
| Modeling | LightGBM, optbinning (WoE scorecard), scikit-learn, lifelines |
| Explainability | SHAP |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Validation | pytest (acceptance checks), matplotlib (PDF report) |
| IFRS 9 ECL | scipy (Vasicek overlay), lifelines (survival term structure), matplotlib (PDF report) |

---

## Notes & limitations

- **Sampled book** — the current run uses a 200K-loan sample of the LendingClub file (≈119K completed loans); portfolio figures are for that sample, not the full 2.2M-loan book.
- **EAD** — the base `EL` table approximates EAD by the loan amount; the **ECL engine improves on this** with a full amortization schedule (declining outstanding balance). A production model would use actual outstanding principal at default.
- **LGD** uses a portfolio-average recovery rate; a fuller model would predict LGD per loan.
- **ECL staging is a proxy** — no origination-vs-now PD history exists in static data, so stages come from absolute PD + delinquency rather than a true SICR test.
- **ECL macro overlay is a prototype** — Vasicek Z-factors are stress assumptions, not fitted from a macro series; the `issue_d` vintage column is now available to support a data-driven upgrade.
- Grade, sub-grade and interest rate are LendingClub's own risk pricing, so they carry high predictive power but partly encode the answer — the more independent signals are FICO, DTI, term, and income.
- The model is a prototype for learning and portfolio purposes, not a production credit decision system.