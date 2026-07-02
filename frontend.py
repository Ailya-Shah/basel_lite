"""
Basel-Lite — Credit Risk Terminal (Streamlit frontend)

Start the backend first (in its own terminal):
    uvicorn backend:app --reload
Then run this:
    streamlit run frontend.py

Config: set BASEL_DB_URL (and optionally BASEL_API_URL) in a .env file at the
project root. Nothing secret is hard-coded here.
"""
import os
from dotenv import load_dotenv
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

load_dotenv()  # read .env from the project's root

API_URL = os.getenv("BASEL_API_URL", "http://127.0.0.1:8000")
DB_URL  = os.getenv("BASEL_DB_URL", "mysql+pymysql://root@localhost:3306/basel_lite")

st.set_page_config(page_title="Basel-Lite · Credit Risk", page_icon="◆", layout="wide")
# ---------------------------------------------------------------------------
# Palette  (cyan/violet brand; risk traffic-light kept for semantics only)
# ---------------------------------------------------------------------------
INK, PANEL, BORDER = "#0A0E1A", "#0F1623", "#1E2A3E"
CYAN, VIOLET = "#22D3EE", "#A78BFA"
GREEN, AMBER, RED = "#34D399", "#FBBF24", "#FB7185"
TEXT, MUTED = "#E6EDF5", "#7C8AA5"

def risk_color(band: str) -> str:
    return {"Low": GREEN, "Medium": AMBER, "High": RED}.get(band, CYAN)

def html(s: str):
    """Inject raw HTML/CSS with blank lines stripped (Streamlit splits HTML
    blocks on blank lines and would otherwise render CSS as text)."""
    st.markdown("\n".join(l for l in s.splitlines() if l.strip()),
                unsafe_allow_html=True)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');
.stApp {{
  background:
    radial-gradient(1100px 520px at 88% -12%, rgba(34,211,238,0.10), transparent 60%),
    radial-gradient(900px 520px at -5% 8%, rgba(167,139,250,0.10), transparent 58%),
    {INK};
  color: {TEXT}; font-family:'Inter',sans-serif;
}}
#MainMenu, footer {{ visibility:hidden; }}
h1,h2,h3,.mono {{ font-family:'JetBrains Mono',monospace; letter-spacing:-0.5px; }}
.hero-title {{ font-size:3.2rem; font-weight:700; margin:0; line-height:1;
  background:linear-gradient(90deg,{CYAN},{VIOLET}); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; }}
.pill {{ display:inline-block; font-family:'JetBrains Mono',monospace; font-size:.72rem;
  letter-spacing:2px; color:{CYAN}; border:1px solid {CYAN}; border-radius:3px;
  padding:5px 12px; margin-top:14px; box-shadow:0 0 18px rgba(34,211,238,0.18) inset; }}
.eyebrow {{ font-family:'JetBrains Mono',monospace; color:{CYAN}; font-size:.72rem;
  letter-spacing:3px; text-transform:uppercase; opacity:.85; }}
.card {{ position:relative; background:linear-gradient(180deg,#101a2b,{PANEL});
  border:1px solid {BORDER}; border-radius:8px; padding:22px 24px; margin-bottom:10px;
  transition:transform .15s ease, border-color .15s ease, box-shadow .15s ease; }}
.card:hover {{ transform:translateY(-2px); border-color:{CYAN};
  box-shadow:0 0 26px rgba(34,211,238,0.12); }}
.card::before,.card::after {{ content:''; position:absolute; width:13px; height:13px;
  border-color:{CYAN}; }}
.card::before {{ top:-1px; left:-1px; border-top:2px solid; border-left:2px solid; }}
.card::after {{ bottom:-1px; right:-1px; border-bottom:2px solid; border-right:2px solid; }}
.metric-label {{ font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:1.5px;
  color:{MUTED}; text-transform:uppercase; }}
.metric-value {{ font-family:'JetBrains Mono',monospace; font-size:2.4rem; font-weight:700;
  line-height:1.1; margin-top:6px; }}
.metric-sub {{ font-size:.78rem; color:{MUTED}; margin-top:4px; }}
.el-readout {{ font-family:'JetBrains Mono',monospace; font-size:3.4rem; font-weight:700;
  line-height:1; }}
.stButton > button {{ background:linear-gradient(90deg,{CYAN},{VIOLET}); color:#06121f;
  border:none; border-radius:6px; font-family:'JetBrains Mono',monospace; font-weight:700;
  letter-spacing:1px; padding:10px 22px; transition:filter .15s ease; }}
.stButton > button:hover {{ filter:brightness(1.12); color:#06121f; }}
section[data-testid="stSidebar"] {{ background:#080C16; border-right:1px solid {BORDER}; }}
hr {{ border-color:{BORDER}; }}
</style>
"""
html(CSS)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.ok else None
    except Exception:
        return None

def api_score(payload):
    r = requests.post(f"{API_URL}/score", json=payload, timeout=10)
    r.raise_for_status(); return r.json()

def api_batch(payloads):
    r = requests.post(f"{API_URL}/score_batch", json=payloads, timeout=90)
    r.raise_for_status(); return r.json()

health = api_health()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
status = (f"<span style='color:{GREEN}'>&#9679; API ONLINE</span>" if health
          else f"<span style='color:{RED}'>&#9679; API OFFLINE &mdash; run: uvicorn backend:app --reload</span>")
html(f"""
<div style="display:flex; justify-content:space-between; align-items:flex-end;">
  <div>
    <div class="hero-title">BASEL&nbsp;&middot;&nbsp;LITE</div>
    <span class="pill">CREDIT RISK &amp; CAPITAL ENGINE</span>
  </div>
  <div class="mono" style="font-size:.8rem; text-align:right;">{status}<br>
    <span style="color:{MUTED}">PD &times; LGD &times; EAD</span></div>
</div>
<hr>
""")

# ---------------------------------------------------------------------------
# Sidebar + option lists
# ---------------------------------------------------------------------------
st.sidebar.markdown("<div class='eyebrow' style='padding:0 0 6px 4px'>// NAVIGATION</div>", unsafe_allow_html=True)
page = st.sidebar.radio("nav", ["Borrower scorer", "Portfolio risk", "Overview"],
                        label_visibility="collapsed")

GRADES = list("ABCDEFG")
SUBGRADES = [g + str(i) for g in GRADES for i in range(1, 6)]
EMP = ["< 1 year","1 year","2 years","3 years","4 years","5 years","6 years",
       "7 years","8 years","9 years","10+ years"]
HOME = ["MORTGAGE","RENT","OWN","OTHER"]
VERIF = ["Verified","Source Verified","Not Verified"]
PURPOSE = ["debt_consolidation","credit_card","home_improvement","major_purchase",
           "medical","car","small_business","vacation","moving","house","other"]
STATES = ["CA","NY","TX","FL","IL","NJ","PA","OH","GA","NC","VA","MI","WA","AZ","MA"]

def metric_card(label, value, sub="", color=TEXT):
    html(f"""<div class="card"><div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color}">{value}</div>
      <div class="metric-sub">{sub}</div></div>""")

def dark_layout(fig, title, h=320):
    fig.update_layout(title=title, height=h, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, family="JetBrains Mono"),
        yaxis=dict(gridcolor=BORDER), xaxis=dict(gridcolor=BORDER),
        margin=dict(l=10, r=10, t=46, b=10))
    return fig

# cache the engine so we don't create a new one on every rerun (Streamlit reloads on every input change)
@st.cache_resource
def get_engine():
    from sqlalchemy import create_engine
    return create_engine(DB_URL)

# ===========================================================================
# PAGE 1 — Borrower scorer (live)
# ===========================================================================
if page == "Borrower scorer":
    html("<div class='eyebrow'>// LIVE BORROWER ASSESSMENT</div>")
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown("**Loan & borrower**")
        fico  = st.slider("FICO score", 600, 850, 690)
        amnt  = st.slider("Loan amount ($)", 1000, 40000, 15000, step=500)
        rate  = st.slider("Interest rate (%)", 5.0, 30.0, 13.0, step=0.1)
        dti   = st.slider("Debt-to-income (%)", 0.0, 45.0, 18.0, step=0.5)
        inc   = st.number_input("Annual income ($)", 10000, 500000, 65000, step=1000)
        c1, c2 = st.columns(2)
        grade = c1.selectbox("Grade", GRADES, index=2)
        term  = c2.selectbox("Term", [" 36 months", " 60 months"])
        with st.expander("Advanced inputs"):
            sub   = st.selectbox("Sub-grade", SUBGRADES, index=SUBGRADES.index(grade + "2"))
            emp   = st.selectbox("Employment length", EMP, index=5)
            home  = st.selectbox("Home ownership", HOME, index=1)
            verif = st.selectbox("Verification", VERIF, index=1)
            purp  = st.selectbox("Purpose", PURPOSE)
            apptype = st.selectbox("Application type", ["Individual", "Joint App"])
            state = st.selectbox("State", STATES)
            inst  = st.number_input("Installment ($)", 20.0, 1500.0, 450.0)
            rev_bal = st.number_input("Revolving balance ($)", 0, 200000, 12000, step=500)
            rev_util = st.slider("Revolving utilisation (%)", 0.0, 130.0, 45.0)
            open_acc = st.number_input("Open accounts", 0, 60, 10)
            total_acc = st.number_input("Total accounts", 0, 120, 25)
            delinq = st.number_input("Delinquencies (2yr)", 0, 20, 0)
            inq = st.number_input("Inquiries (6mo)", 0, 20, 1)
            pub = st.number_input("Public records", 0, 10, 0)
            mort = st.number_input("Mortgage accounts", 0, 20, 1)
            bankr = st.number_input("Bankruptcies", 0, 10, 0)

    payload = {
        "loan_amnt": amnt, "term": term, "int_rate": rate, "installment": inst,
        "grade": grade, "sub_grade": sub, "emp_length": emp, "home_ownership": home,
        "annual_inc": inc, "verification_status": verif, "purpose": purp, "dti": dti,
        "delinq_2yrs": delinq, "inq_last_6mths": inq, "open_acc": open_acc, "pub_rec": pub,
        "revol_bal": rev_bal, "revol_util": rev_util, "total_acc": total_acc,
        "application_type": apptype, "mort_acc": mort, "pub_rec_bankruptcies": bankr,
        "addr_state": state, "fico_score": fico,
    }

    with right:
        if not health:
            st.warning("Backend offline. Start it with `uvicorn backend:app --reload`, then reload.")
        else:
            try:
                res = api_score(payload)              # scores live on every change
                col = risk_color(res["risk_band"])
                m1, m2 = st.columns(2)
                with m1:
                    metric_card("Probability of default",
                                f"{res['probability_of_default']*100:.1f}%",
                                "book average &#8776; 20%", col)
                with m2:
                    metric_card("Risk band", res["risk_band"].upper(),
                                f"LGD {res['lgd']*100:.0f}%", col)
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=res["credit_score"],
                    number={"font": {"color": col, "size": 40, "family": "JetBrains Mono"}},
                    gauge={"axis": {"range": [300, 850], "tickcolor": MUTED},
                           "bar": {"color": col, "thickness": 0.28},
                           "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                           "steps": [{"range": [300, 580], "color": "rgba(251,113,133,.16)"},
                                     {"range": [580, 670], "color": "rgba(251,191,36,.16)"},
                                     {"range": [670, 850], "color": "rgba(52,211,153,.16)"}]}))
                gauge.update_layout(height=230, margin=dict(l=20, r=20, t=8, b=0),
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color=TEXT, family="JetBrains Mono"))
                html("<div class='metric-label'>CREDIT SCORE (300&ndash;850)</div>")
                st.plotly_chart(gauge, use_container_width=True)
                html(f"""<div class="card">
                  <div class="metric-label">EXPECTED LOSS ON THIS LOAN</div>
                  <div class="el-readout" style="color:{col};text-shadow:0 0 22px {col}55">
                    ${res['expected_loss']:,.0f}</div>
                  <div class="metric-sub">PD {res['probability_of_default']*100:.1f}% &times;
                    LGD {res['lgd']*100:.0f}% &times; EAD ${res['ead']:,.0f}</div></div>""")
            except Exception as e:
                st.error(f"Scoring failed: {e}")

# ===========================================================================
# PAGE 2 — Portfolio risk
# ===========================================================================
elif page == "Portfolio risk":
    html("<div class='eyebrow'>// PORTFOLIO EXPECTED LOSS</div>")
    n = st.slider("Loans to sample from the book", 100, 3000, 800, step=100)
    run = st.button("RUN PORTFOLIO")
    if run:
        if not health:
            st.warning("Backend offline. Start it with `uvicorn backend:app --reload`.")
        else:
            try:
                eng = get_engine()
                df = pd.read_sql(f"SELECT * FROM loans_clean ORDER BY RAND() LIMIT {n}", eng)
                df = df.drop(columns=[c for c in ["default"] if c in df.columns])
                # JSON can't carry NaN/inf — fill numerics (median), categoricals ("n/a")
                df = df.replace([np.inf, -np.inf], np.nan)
                num = df.select_dtypes(include="number").columns
                obj = df.select_dtypes(include="object").columns
                df[num] = df[num].fillna(df[num].median(numeric_only=True)).fillna(0)
                df[obj] = df[obj].fillna("n/a")
                payloads = json.loads(df.to_json(orient="records"))
                res = api_batch(payloads)

                c1, c2, c3, c4 = st.columns(4)
                with c1: metric_card("Total exposure", f"${res['total_ead']/1e6:.2f}M", "EAD", TEXT)
                with c2: metric_card("Expected loss", f"${res['total_expected_loss']/1e6:.2f}M",
                                     "capital to reserve", VIOLET)
                with c3: metric_card("Loss rate", f"{res['expected_loss_rate']*100:.2f}%",
                                     "of exposure", RED)
                avg_pd = sum(r["pd"] for r in res["results"]) / len(res["results"])
                with c4: metric_card("Average PD", f"{avg_pd*100:.1f}%", f"{res['n']} loans", CYAN)

                out = pd.DataFrame(res["results"]); out["grade"] = df["grade"].values
                by_grade = out.groupby("grade")["expected_loss"].sum().reindex(GRADES).fillna(0)
                fig = go.Figure(go.Bar(x=by_grade.index, y=by_grade.values,
                    marker=dict(color=by_grade.values, colorscale=[[0, CYAN], [1, VIOLET]])))
                st.plotly_chart(dark_layout(fig, "Expected loss by grade"), use_container_width=True)
                fig2 = go.Figure(go.Histogram(x=out["pd"], nbinsx=30, marker_color=CYAN))
                st.plotly_chart(dark_layout(fig2, "Distribution of default probability", 300),
                                use_container_width=True)
            except Exception as e:
                st.error(f"Could not load portfolio: {e}")
    else:
        st.info("Pick a sample size and press **RUN PORTFOLIO** to value a sampled slice of the book.")

# ===========================================================================
# PAGE 3 — Overview
# ===========================================================================
else:
    html("<div class='eyebrow'>// PROJECT OVERVIEW</div>")
    a, b, c = st.columns(3)
    with a:
        html(f"""<div class="card"><div class="metric-label">// WHAT IT DOES</div>
        <p style="color:{MUTED};line-height:1.7;margin-top:14px">Estimates the
        <b style="color:{TEXT}">probability of default</b>, assigns a
        <b style="color:{TEXT}">300&ndash;850 credit score</b>, and computes the
        <b style="color:{TEXT}">expected loss</b> for any borrower &mdash; loan by loan
        and across a sampled book.</p></div>""")
    with b:
        html(f"""<div class="card"><div class="metric-label">// THE MODEL</div>
        <p style="color:{MUTED};line-height:1.7;margin-top:14px">WoE scorecard +
        calibrated <b style="color:{TEXT}">LightGBM</b> PD model, LGD measured from real
        recoveries, assembled as <b style="color:{CYAN}">EL = PD &times; LGD &times; EAD</b>.
        </p></div>""")
    with c:
        html(f"""<div class="card"><div class="metric-label">// DATA</div>
        <p style="color:{MUTED};line-height:1.7;margin-top:14px">LendingClub 2007&ndash;2018,
        application-time features only (no leakage), served from
        <b style="color:{TEXT}">MySQL</b> through a
        <b style="color:{TEXT}">FastAPI</b> backend.</p></div>""")