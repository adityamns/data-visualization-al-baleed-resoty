#!/bin/bash
set -e

APP_DIR="/var/data-visualization-al-baleed-resoty"
VENV="$APP_DIR/venv"

echo "========================================"
echo "  Al Baleed Resort Dashboard Launcher"
echo "========================================"
echo

echo "Starting dashboard..."
$VENV/bin/streamlit run streamlit_dashboard.py \
  --server.address 0.0.0.0 \
  --server.port 8501
