"""
app.py
------
Loan Officer Dashboard — Domino App (Dash)

Provides a form-based UI for loan officers to:
  - Enter loan application details
  - Submit to the deployed Domino Model API endpoint
  - View default probability, risk tier, recommendation
  - Inspect a SHAP waterfall chart explaining the decision (if SHAP installed)

Environment variables required:
    DOMINO_MODEL_API_URL    : Full URL of the deployed Domino endpoint
    DOMINO_MODEL_API_KEY    : API key for the endpoint

Usage:
    python app/app.py
"""

import os
import json
import logging

import requests
import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domino endpoint config
# ---------------------------------------------------------------------------
MODEL_API_URL = os.environ.get("DOMINO_MODEL_API_URL", "")
MODEL_API_KEY = os.environ.get("DOMINO_MODEL_API_KEY", "")

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
runurl = os.environ.get("DOMINO_RUN_HOST_PATH")

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    routes_pathname_prefix="/",
    requests_pathname_prefix=runurl,
    title="Loan Officer Dashboard",
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "Low":    "#2ecc71",
    "Medium": "#f39c12",
    "High":   "#e74c3c",
    "bg":     "#f8f9fa",
    "card":   "#ffffff",
    "text":   "#2c3e50",
    "muted":  "#7f8c8d",
}

TIER_BADGE = {
    "Low":    "success",
    "Medium": "warning",
    "High":   "danger",
}


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def input_row(label, id_, type_="number", value=None, options=None, placeholder=""):
    if options:
        control = dbc.Select(
            id=id_,
            options=[{"label": o, "value": o} for o in options],
            value=value,
        )
    else:
        control = dbc.Input(
            id=id_, type=type_, value=value, placeholder=placeholder,
            debounce=True,
        )
    return dbc.Row([
        dbc.Label(label, width=5, style={"fontWeight": "500", "fontSize": "0.88rem"}),
        dbc.Col(control, width=7),
    ], className="mb-2 align-items-center")


def metric_card(title, value_id, subtitle="", color="#2c3e50"):
    return dbc.Card([
        dbc.CardBody([
            html.P(title, className="text-muted mb-1",
                   style={"fontSize": "0.8rem", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.H3(id=value_id, children="—",
                    style={"fontWeight": "700", "color": color}),
            html.P(subtitle, className="text-muted mb-0",
                   style={"fontSize": "0.78rem"}),
        ])
    ], style={"borderRadius": "10px", "border": "none",
              "boxShadow": "0 2px 8px rgba(0,0,0,0.07)"})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = dbc.Container(fluid=True, style={"backgroundColor": COLORS["bg"], "minHeight": "100vh"}, children=[

    # Header
    dbc.Row([
        dbc.Col([
            html.H2("🏦 Loan Officer Dashboard",
                    style={"fontWeight": "700", "color": COLORS["text"], "marginBottom": "4px"}),
            html.P("LendingClub Credit Risk Scoring — Powered by Domino",
                   style={"color": COLORS["muted"], "fontSize": "0.9rem"}),
        ])
    ], className="pt-4 pb-2 px-4"),

    dbc.Row([

        # ── LEFT PANEL: Input form ──────────────────────────────────────────
        dbc.Col(width=3, children=[
            dbc.Card(style={"borderRadius": "12px", "border": "none",
                            "boxShadow": "0 2px 12px rgba(0,0,0,0.08)"}, children=[
                dbc.CardHeader(html.B("Loan Application", style={"fontSize": "0.95rem"})),
                dbc.CardBody([

                    html.P("Loan Details", className="text-muted mb-2",
                           style={"fontSize": "0.78rem", "textTransform": "uppercase"}),
                    input_row("Loan Amount ($)",    "inp-loan-amnt",    value=15000),
                    input_row("Interest Rate (%)",  "inp-int-rate",     value=13.5),
                    input_row("Term (months)",      "inp-term",
                              options=["36", "60"], value="36"),
                    input_row("Purpose",            "inp-purpose",
                              options=["debt_consolidation", "credit_card", "home_improvement",
                                       "major_purchase", "small_business", "medical",
                                       "moving", "vacation", "wedding", "other"],
                              value="debt_consolidation"),
                    input_row("Grade",              "inp-grade",
                              options=["A", "B", "C", "D", "E", "F", "G"], value="C"),

                    html.Hr(),
                    html.P("Applicant Details", className="text-muted mb-2",
                           style={"fontSize": "0.78rem", "textTransform": "uppercase"}),
                    input_row("Annual Income ($)",  "inp-annual-inc",   value=65000),
                    input_row("DTI (%)",            "inp-dti",          value=18.5),
                    input_row("Home Ownership",     "inp-home",
                              options=["RENT", "MORTGAGE", "OWN", "OTHER"],
                              value="RENT"),
                    input_row("Verification",       "inp-verification",
                              options=["Not Verified", "Source Verified", "Verified"],
                              value="Verified"),

                    html.Hr(),
                    html.P("Credit Profile", className="text-muted mb-2",
                           style={"fontSize": "0.78rem", "textTransform": "uppercase"}),
                    input_row("Monthly Installment ($)", "inp-installment",  value=350),
                    input_row("Revolving Balance ($)",   "inp-revol-bal",    value=12000),
                    input_row("Revolving Util (%)",      "inp-revol-util",   value=55),
                    input_row("Open Accounts",           "inp-open-acc",     value=7),
                    input_row("Total Accounts",          "inp-total-acc",    value=18),
                    input_row("Public Records",          "inp-pub-rec",      value=0),
                    input_row("Delinquencies (2yr)",     "inp-delinq",       value=0),

                    html.Br(),
                    dbc.Button("Score Application", id="btn-score", color="primary",
                               className="w-100", size="lg", n_clicks=0),

                    # Error alert
                    dbc.Alert(id="alert-error", is_open=False, color="danger",
                              className="mt-3 mb-0", style={"fontSize": "0.85rem"}),
                ])
            ])
        ]),

        # ── RIGHT PANEL: Results ────────────────────────────────────────────
        dbc.Col(width=9, children=[

            # Metric cards row
            dbc.Row([
                dbc.Col(metric_card("Default Probability", "out-prob",
                                    "Likelihood of default"), width=3),
                dbc.Col(metric_card("Risk Score",          "out-score",
                                    "100 = safest, 0 = riskiest"), width=3),
                dbc.Col(metric_card("Risk Tier",           "out-tier"), width=3),
                dbc.Col(metric_card("Recommendation",      "out-rec"), width=3),
            ], className="mb-4"),

            # Gauge + SHAP row
            dbc.Row([
                # Gauge
                dbc.Col(width=5, children=[
                    dbc.Card(style={"borderRadius": "12px", "border": "none",
                                    "boxShadow": "0 2px 12px rgba(0,0,0,0.08)"}, children=[
                        dbc.CardHeader(html.B("Default Probability", style={"fontSize": "0.95rem"})),
                        dbc.CardBody([
                            dcc.Graph(id="gauge-chart", config={"displayModeBar": False},
                                      style={"height": "280px"}),
                        ])
                    ])
                ]),

                # SHAP waterfall
                dbc.Col(width=7, children=[
                    dbc.Card(style={"borderRadius": "12px", "border": "none",
                                    "boxShadow": "0 2px 12px rgba(0,0,0,0.08)"}, children=[
                        dbc.CardHeader(html.B("Decision Drivers (SHAP)", style={"fontSize": "0.95rem"})),
                        dbc.CardBody([
                            dcc.Graph(id="shap-chart", config={"displayModeBar": False},
                                      style={"height": "280px"}),
                        ])
                    ])
                ]),
            ], className="mb-4"),

            # Raw response
            dbc.Row([
                dbc.Col([
                    dbc.Card(style={"borderRadius": "12px", "border": "none",
                                    "boxShadow": "0 2px 12px rgba(0,0,0,0.08)"}, children=[
                        dbc.CardHeader(html.B("Raw API Response", style={"fontSize": "0.95rem"})),
                        dbc.CardBody([
                            html.Pre(id="raw-response",
                                     style={"fontSize": "0.82rem", "maxHeight": "180px",
                                            "overflowY": "auto", "backgroundColor": "#f4f6f8",
                                            "padding": "12px", "borderRadius": "8px",
                                            "color": COLORS["text"]}),
                        ])
                    ])
                ])
            ]),

        ]),
    ], className="px-4 pb-4"),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("out-prob",      "children"),
    Output("out-score",     "children"),
    Output("out-tier",      "children"),
    Output("out-tier",      "style"),
    Output("out-rec",       "children"),
    Output("out-rec",       "style"),
    Output("gauge-chart",   "figure"),
    Output("shap-chart",    "figure"),
    Output("raw-response",  "children"),
    Output("alert-error",   "children"),
    Output("alert-error",   "is_open"),
    Input("btn-score", "n_clicks"),
    State("inp-loan-amnt",    "value"),
    State("inp-int-rate",     "value"),
    State("inp-term",         "value"),
    State("inp-purpose",      "value"),
    State("inp-grade",        "value"),
    State("inp-annual-inc",   "value"),
    State("inp-dti",          "value"),
    State("inp-home",         "value"),
    State("inp-verification", "value"),
    State("inp-installment",  "value"),
    State("inp-revol-bal",    "value"),
    State("inp-revol-util",   "value"),
    State("inp-open-acc",     "value"),
    State("inp-total-acc",    "value"),
    State("inp-pub-rec",      "value"),
    State("inp-delinq",       "value"),
    prevent_initial_call=True,
)
def score_application(n_clicks, loan_amnt, int_rate, term, purpose, grade,
                       annual_inc, dti, home, verification, installment,
                       revol_bal, revol_util, open_acc, total_acc, pub_rec, delinq):

    empty_gauge = _empty_gauge()
    empty_shap  = _empty_shap()
    defaults    = ("—", "—", "—", {}, "—", {}, empty_gauge, empty_shap, "", "", False)

    if not n_clicks:
        return defaults

    # Build request payload
    payload = {
        "data": {
            "loan_amnt":           float(loan_amnt   or 0),
            "int_rate":            float(int_rate    or 0),
            "term":                str(term          or "36"),
            "purpose":             str(purpose       or "other"),
            "grade":               str(grade         or "A"),
            "annual_inc":          float(annual_inc  or 0),
            "dti":                 float(dti         or 0),
            "home_ownership":      str(home          or "RENT"),
            "verification_status": str(verification  or "Not Verified"),
            "installment":         float(installment or 0),
            "revol_bal":           float(revol_bal   or 0),
            "revol_util":          float(revol_util  or 0),
            "open_acc":            int(open_acc      or 0),
            "total_acc":           int(total_acc     or 0),
            "pub_rec":             int(pub_rec       or 0),
            "delinq_2yrs":         int(delinq        or 0),
        }
    }

    # Call endpoint
    try:
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {MODEL_API_KEY}",
        }
        response = requests.post(
            MODEL_API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=15,
        )
        response.raise_for_status()
        result = response.json().get("result", response.json())

    except requests.exceptions.ConnectionError:
        err = "Could not connect to the model endpoint. Check DOMINO_MODEL_API_URL."
        return ("—", "—", "—", {}, "—", {}, empty_gauge, empty_shap, "", err, True)
    except requests.exceptions.Timeout:
        err = "Request timed out. The endpoint may be starting up — try again."
        return ("—", "—", "—", {}, "—", {}, empty_gauge, empty_shap, "", err, True)
    except Exception as e:
        err = f"Unexpected error: {str(e)}"
        return ("—", "—", "—", {}, "—", {}, empty_gauge, empty_shap, "", err, True)

    # Parse result
    prob       = result.get("default_probability", 0)
    risk_score = result.get("risk_score", 0)
    tier       = result.get("risk_tier", "—")
    rec        = result.get("recommendation", "—")
    shap_vals  = result.get("shap_values", {})

    tier_color = COLORS.get(tier, COLORS["text"])
    tier_style = {"color": tier_color, "fontWeight": "700"}
    rec_style  = {"color": tier_color, "fontWeight": "700"}

    gauge = _build_gauge(prob)
    shap  = _build_shap_chart(shap_vals)
    raw   = json.dumps(result, indent=2)

    return (
        f"{prob:.1%}", str(risk_score), tier, tier_style,
        rec, rec_style, gauge, shap, raw, "", False
    )


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

    colors = [COLORS["High"] if v > 0 else COLORS["Low"] for v in values]
    labels = [f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values]

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
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#bdc3c7"},
               "steps": [{"range": [0, 100], "color": "#ecf0f1"}]},
    ))
    fig.update_layout(margin=dict(t=30, b=10, l=20, r=20), paper_bgcolor="white")
    return fig


def _empty_shap() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{"text": "Score an application to see decision drivers",
                       "showarrow": False, "font": {"color": COLORS["muted"], "size": 13},
                       "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}],
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=False)