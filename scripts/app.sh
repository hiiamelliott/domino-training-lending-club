#!/usr/bin/env bash
# ==============================================================================
# app.sh — Domino App launcher for the Loan Officer Dashboard (Streamlit)
#
# Required environment variables (set in Domino App settings):
#   DOMINO_MODEL_API_URL  : Full URL of the deployed Domino scoring endpoint
#   DOMINO_MODEL_API_KEY  : API key for the endpoint (Find in your Account settings)
# ==============================================================================

set -e

echo "Starting Loan Officer Dashboard (Streamlit)..."
echo "Model API URL: ${DOMINO_MODEL_API_URL:-'NOT SET — update in App environment variables'}"

pip install streamlit plotly requests numpy pandas shap --quiet

# DOMINO_RUN_HOST_PATH is set automatically by Domino and provides the base URL
# path for the app (e.g. /api/runs/<id>/). Streamlit needs this to serve assets
# correctly when running behind Domino's reverse proxy.
BASE_PATH="${DOMINO_RUN_HOST_PATH:-/}"

streamlit run scripts/app.py \
    --server.port 8888 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.baseUrlPath "${BASE_PATH}"
