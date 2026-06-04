#!/usr/bin/env bash
# ==============================================================================
# app.sh — Domino App launcher for the Loan Officer Dashboard (Dash)
#
# Required environment variables (set in Domino App settings):
#   DOMINO_MODEL_API_URL  : Full URL of the deployed Domino scoring endpoint
#   DOMINO_MODEL_API_KEY  : API key for the endpoint (Find in your Account settings)
# ==============================================================================

set -e

echo "Starting Loan Officer Dashboard..."
echo "Model API URL: ${DOMINO_MODEL_API_URL:-'NOT SET — update in App environment variables'}"
pip install dash dash-bootstrap-components plotly requests --quiet

# Launch Dash app
python3 scripts/app.py