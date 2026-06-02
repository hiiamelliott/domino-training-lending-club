"""
app.py
------
Loan Officer Dashboard — Domino App (Streamlit)

Provides a form-based UI for loan officers to:
  - Enter loan application details
  - Submit to the deployed Domino Model API endpoint
  - View default probability, risk tier, recommendation
  - Inspect a SHAP waterfall chart explaining the decision (if model returns shap_values)

Environment variables required:
    DOMINO_MODEL_API_URL    : Full URL of the deployed Domino endpoint
    DOMINO_MODEL_API_KEY    : API key for the endpoint

Usage:
    streamlit run scripts/app.py --server.port 8888 --server.address 0.0.0.0
"""

import json
import logging
import os

import plotly.graph_objects as go
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_API_URL = os.environ.get("DOMINO_MODEL_API_URL", "")
MODEL_API_KEY = os.environ.get("DOMINO_MODEL_API_KEY", "")

COLORS = {
    "Low":    "#2ecc71",
    "Medium": "#f39c12",
    "High":   "#e74c3c",
    "text":   "#2c3e50",
    "muted":  "#7f8c8d",
}


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _build_gauge(prob: float) -> go.Figure:
    color = (COLORS["Low"] if prob < 0.15
             else COLORS["Medium"] if prob < 0.35
             else COLORS["High"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar":  {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,  15], "color": "#d5f5e3"},
                {"range": [15, 35], "color": "#fdebd0"},
                {"range": [35, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="white",
        font={"family": "sans-serif"},
    )
    return fig


def _build_shap_chart(shap_vals: dict) -> go.Figure:
    if not shap_vals:
        return _empty_shap()

    features = list(shap_vals.keys())
    values   = list(shap_vals.values())
    colors   = [COLORS["High"] if v > 0 else COLORS["Low"] for v in values]
    labels   = [f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        title={"text": "Red = increases default risk  |  Green = decreases",
               "font": {"size": 11, "color": COLORS["muted"]}},
        xaxis_title="SHAP Value",
        margin=dict(t=40, b=20, l=10, r=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "sans-serif", "size": 11},
        yaxis={"autorange": "reversed"},
    )
    return fig


def _empty_gauge() -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0,
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#bdc3c7"},
            "steps": [{"range": [0, 100], "color": "#ecf0f1"}],
        },
    ))
    fig.update_layout(margin=dict(t=30, b=10, l=20, r=20), paper_bgcolor="white")
    return fig


def _empty_shap() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{
            "text": "Score an application to see decision drivers",
            "showarrow": False,
            "font": {"color": COLORS["muted"], "size": 13},
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
        }],
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Loan Officer Dashboard",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Loan Officer Dashboard")
st.caption("LendingClub Credit Risk Scoring — Powered by Domino")

# ---------------------------------------------------------------------------
# Sidebar — input form
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Loan Application")

    st.markdown("**Loan Details**")
    loan_amnt   = st.number_input("Loan Amount ($)",     min_value=0.0, value=15000.0, step=500.0)
    int_rate    = st.number_input("Interest Rate (%)",   min_value=0.0, max_value=100.0, value=13.5, step=0.1)
    term        = st.selectbox("Term (months)", ["36", "60"])
    purpose     = st.selectbox("Purpose", [
        "debt_consolidation", "credit_card", "home_improvement",
        "major_purchase", "small_business", "medical",
        "moving", "vacation", "wedding", "other",
    ])
    grade       = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"], index=2)

    st.markdown("---")
    st.markdown("**Applicant Details**")
    annual_inc  = st.number_input("Annual Income ($)",   min_value=0.0, value=65000.0, step=1000.0)
    dti         = st.number_input("DTI (%)",             min_value=0.0, max_value=100.0, value=18.5, step=0.1)
    home        = st.selectbox("Home Ownership",         ["RENT", "MORTGAGE", "OWN", "OTHER"])
    verification = st.selectbox("Verification Status",  ["Not Verified", "Source Verified", "Verified"], index=2)

    st.markdown("---")
    st.markdown("**Credit Profile**")
    installment = st.number_input("Monthly Installment ($)", min_value=0.0, value=350.0,   step=10.0)
    revol_bal   = st.number_input("Revolving Balance ($)",   min_value=0.0, value=12000.0, step=500.0)
    revol_util  = st.number_input("Revolving Util (%)",      min_value=0.0, max_value=100.0, value=55.0, step=0.5)
    open_acc    = st.number_input("Open Accounts",           min_value=0,   value=7,  step=1)
    total_acc   = st.number_input("Total Accounts",          min_value=0,   value=18, step=1)
    pub_rec     = st.number_input("Public Records",          min_value=0,   value=0,  step=1)
    delinq      = st.number_input("Delinquencies (2yr)",     min_value=0,   value=0,  step=1)

    st.markdown("---")
    submitted = st.button("Score Application", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Call the model endpoint on submission
# ---------------------------------------------------------------------------

if submitted:
    if not MODEL_API_URL:
        st.session_state["error"]  = "DOMINO_MODEL_API_URL is not set. Add it in the App environment variables."
        st.session_state["result"] = None
    else:
        payload = {
            "data": {
                "loan_amnt":           float(loan_amnt),
                "int_rate":            float(int_rate),
                "term":                str(term),
                "purpose":             str(purpose),
                "grade":               str(grade),
                "annual_inc":          float(annual_inc),
                "dti":                 float(dti),
                "home_ownership":      str(home),
                "verification_status": str(verification),
                "installment":         float(installment),
                "revol_bal":           float(revol_bal),
                "revol_util":          float(revol_util),
                "open_acc":            int(open_acc),
                "total_acc":           int(total_acc),
                "pub_rec":             int(pub_rec),
                "delinq_2yrs":         int(delinq),
            }
        }
        try:
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {MODEL_API_KEY}",
            }
            log.info("Scoring application via %s", MODEL_API_URL)
            response = requests.post(
                MODEL_API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=15,
            )
            response.raise_for_status()
            st.session_state["result"] = response.json().get("result", response.json())
            st.session_state["error"]  = None

        except requests.exceptions.ConnectionError:
            st.session_state["error"]  = "Could not connect to the model endpoint. Check DOMINO_MODEL_API_URL."
            st.session_state["result"] = None
        except requests.exceptions.Timeout:
            st.session_state["error"]  = "Request timed out. The endpoint may be starting up — try again."
            st.session_state["result"] = None
        except Exception as exc:
            st.session_state["error"]  = f"Unexpected error: {exc}"
            st.session_state["result"] = None

# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------

error  = st.session_state.get("error")
result = st.session_state.get("result")

if error:
    st.error(error)

# Metrics row
prob       = result.get("default_probability", None) if result else None
risk_score = result.get("risk_score",          None) if result else None
tier       = result.get("risk_tier",           "—")  if result else "—"
rec        = result.get("recommendation",      "—")  if result else "—"
shap_vals  = result.get("shap_values",         {})   if result else {}

col1, col2, col3, col4 = st.columns(4)
col1.metric("Default Probability", f"{prob:.1%}"  if prob       is not None else "—")
col2.metric("Risk Score",          str(risk_score) if risk_score is not None else "—")
col3.metric("Risk Tier",           tier)
col4.metric("Recommendation",      rec)

st.markdown("---")

# Charts row
col_gauge, col_shap = st.columns([2, 3])
with col_gauge:
    st.subheader("Default Probability")
    fig_gauge = _build_gauge(prob) if prob is not None else _empty_gauge()
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_shap:
    st.subheader("Decision Drivers (SHAP)")
    st.plotly_chart(_build_shap_chart(shap_vals), use_container_width=True)

# Raw response
if result:
    with st.expander("Raw API Response"):
        st.code(json.dumps(result, indent=2), language="json")
